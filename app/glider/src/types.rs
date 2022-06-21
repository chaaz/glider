//! The type system and related math for our inference compiler.

use crate::callable::{FnDef, NativeDef};
use crate::custom::{Custom, CustomType};
use crate::parser::UseCount;
use crate::pick;
use crate::value::Value;
use std::hash::{Hash, Hasher};

const MAX_CONST_ELEMENTS: usize = 10;

#[derive(Clone, Debug)]
pub enum Type<C: Custom> {
  Unit,
  Number(ConstNumber),
  Bool(Option<bool>),
  String(Option<String>),
  Array(Box<Array<C>>),
  Object(Box<Object<C>>),
  Json,
  FnDef(FnDef<C>),
  Native(NativeDef<C>),
  Iter(Box<Type<C>>),
  Custom(C::Type),

  /// This type is used for first-pass compilation of a function (at its call site) in order to determine its
  /// type: it's possible that some expressions in recursive functions can't be known or compiled until the
  /// calling function type is first known.
  Unknown
}

impl<C: Custom> UseCount for Type<C> {
  // TODO: Remove this "SINGLE" hack for single use.
  fn is_single_use(&self) -> bool {
    match self {
      Self::String(Some(v)) if v == "SINGLE" => true,
      Self::FnDef(f) => f.captures().iter().any(|(_, c)| c.is_single_use()),
      Self::Native(f) => f.captures().iter().any(|(_, c)| c.is_single_use()),
      Self::Array(a) => a.is_single_use(),
      Self::Object(o) => o.is_single_use(),
      Self::Iter(t) => t.is_single_use(),
      _ => false
    }
  }
}

impl<C: Custom> Default for Type<C> {
  fn default() -> Type<C> { Type::Bool(None) }
}

// TODO: strip constexpr from types when doing a fn_impl or native_impl lookup, to be consistent with
// `PartialEq`.

impl<C: Custom> PartialEq for Type<C> {
  fn eq(&self, other: &Self) -> bool {
    match (self, other) {
      (Self::Unit, Self::Unit) => true,
      (Self::Number(_), Self::Number(_)) => true,
      (Self::Bool(_), Self::Bool(_)) => true,
      (Self::String(_), Self::String(_)) => true,
      (Self::Json, Self::Json) => true,
      (Self::Object(a), Self::Object(b)) => a == b,
      (Self::Array(a), Self::Array(b)) => a == b,
      (Self::FnDef(a), Self::FnDef(b)) => a == b,
      (Self::Iter(a), Self::Iter(b)) => a == b,
      (Self::Unknown, Self::Unknown) => true,
      (Self::Custom(a), Self::Custom(b)) => a == b,
      _ => false
    }
  }
}

impl<C: Custom> Eq for Type<C> {}

impl<C: Custom> Hash for Type<C> {
  fn hash<H: Hasher>(&self, state: &mut H) {
    state.write_usize(23810);
    match self {
      Self::Unit => state.write_usize(4392),
      Self::Number(_) => state.write_usize(1389361),
      Self::Bool(_) => state.write_usize(9744235),
      Self::String(_) => state.write_usize(21284),
      Self::Json => state.write_usize(673727),
      Self::Object(v) => v.hash(state),
      Self::Array(v) => v.hash(state),
      Self::FnDef(v) => v.hash(state),
      Self::Iter(v) => {
        state.write_usize(1283023);
        v.hash(state)
      }
      Self::Native(v) => v.hash(state),
      Self::Custom(v) => v.hash(state),
      Self::Unknown => state.write_usize(12841)
    }
  }
}

impl<C: Custom> Type<C> {
  pub fn is_object(&self) -> bool { matches!(self, Type::Object(_)) }
  pub fn is_array(&self) -> bool { matches!(self, Type::Array(_)) }
  pub fn is_json(&self) -> bool { matches!(self, Self::Json) }
  pub fn is_string(&self) -> bool { matches!(self, Self::String(_)) }
  pub fn is_string_literal(&self) -> bool { matches!(self, Self::String(Some(_))) }
  pub fn is_number(&self) -> bool { matches!(self, Self::Number(_)) }
  pub fn is_bool(&self) -> bool { matches!(self, Self::Bool(_)) }
  pub fn is_fn(&self) -> bool { matches!(self, Self::FnDef(_)) }
  pub fn is_unknown(&self) -> bool { matches!(self, Type::Unknown) }
  pub fn is_iter(&self) -> bool { matches!(self, Type::Iter(_)) }

  pub fn as_array(&self) -> &Array<C> { pick!(self, Self::Array(a) => a, "Not an array: {:?}") }
  pub fn as_object(&self) -> &Object<C> { pick!(self, Self::Object(o) => o, "Not an object: {:?}") }
  pub fn as_string_literal(&self) -> &str { pick!(self, Self::String(Some(s)) => s, "Not a string literal: {:?}") }
  pub fn as_string(&self) -> &Option<String> { pick!(self, Self::String(s) => s, "Not a string type: {:?}") }
  pub fn as_fn(&self) -> &FnDef<C> { pick!(self, Self::FnDef(d) => d, "Not a function: {:?}") }
  pub fn as_native(&self) -> &NativeDef<C> { pick!(self, Self::Native(v) => v, "Not a native: {:?}") }
  pub fn into_object(self) -> Box<Object<C>> { pick!(self, Self::Object(o) => o, "Not an object: {:?}") }
  pub fn into_array(self) -> Box<Array<C>> { pick!(self, Self::Array(a) => a, "Not an array: {:?}") }
  pub fn as_iter(&self) -> &Type<C> { pick!(self, Self::Iter(t) => t, "Not an iter: {:?}") }

  /// Create a type for a function that cannot be currently analyzed, because it is part of a pending recursive
  /// chain.
  pub fn recursion() -> Type<C> {
    // This is the only place that Type::Unknown should be created; in all other cases, the type should be
    // known.
    Type::Unknown
  }

  /// The total number of atomic elements in the type.
  ///
  /// This is `1` for non-compound types, and the total flat length for array and object types.
  pub fn flat_len(&self) -> usize {
    match self {
      Self::Array(a) => a.flat_len(),
      Self::Object(o) => o.flat_len(),
      _ => 1
    }
  }

  /// Combine multiple type options into a single type.
  ///
  /// This returns the common known type that is available
  pub fn unify(types: &[Type<C>]) -> Type<C> {
    let unified = types.iter().fold(Type::Unknown, |c, t| {
      if c.is_unknown() && t.is_unknown() {
        Type::Unknown
      } else if &c == t || t.is_unknown() {
        c
      } else {
        // Don't allow mixed types.
        assert!(c.is_unknown());
        t.clone()
      }
    });

    unified
  }

  /// Get the constexpr value of the type, if one exists.
  ///
  /// For now, the type of a function is the function itself (different functions are considered different
  /// types), so this will return `None` on function types; use `as_fn` instead. Since JSON types don't
  /// participate in the type system, there is no such thing as a constexpr JSON value, so this will return
  /// `None` for JSON types.
  pub fn to_constexpr(&self) -> Option<Value<C>> {
    match self {
      Self::Unit => Some(Value::Unit),
      Self::Number(v) => v.to_value(),
      Self::Bool(v) => v.map(Value::Bool),
      Self::String(v) => v.as_deref().map(|v| v.into()),
      Self::Array(a) => Some(Value::Array(a.types().iter().map(|v| v.to_constexpr().unwrap_or(Value::Void)).collect())),
      Self::Object(o) => Some(Value::Array(o.ordered().map(|v| v.to_constexpr().unwrap_or(Value::Void)).collect())),
      Self::Json => None,
      Self::FnDef(_) => None,
      Self::Native(_) => None,
      Self::Custom(_) => None,
      Self::Iter(_) => None,
      Self::Unknown => None
    }
  }

  /// Detects if two types are represented by the same code execution. Very similar to `Eq`, except that
  /// functions are compared based on their source code position, instead of pointer.
  pub fn similar(&self, other: &Self) -> bool {
    match (self, other) {
      (Self::Unit, Self::Unit) => true,
      (Self::Number(_), Self::Number(_)) => true,
      (Self::Bool(_), Self::Bool(_)) => true,
      (Self::String(_), Self::String(_)) => true,
      (Self::Json, Self::Json) => true,
      (Self::Object(a), Self::Object(b)) => a.similar(b),
      (Self::Array(a), Self::Array(b)) => a.similar(b),
      (Self::FnDef(a), Self::FnDef(b)) => a.similar(b),
      (Self::Native(a), Self::Native(b)) => a.similar(b),
      (Self::Custom(a), Self::Custom(b)) => a.similar(b),
      (Self::Iter(a), Self::Iter(b)) => a.similar(b),
      (Self::Unknown, Self::Unknown) => true,
      _ => false
    }
  }
}

/// The constexpr of a numeric type.
#[derive(Clone, Debug)]
pub enum ConstNumber {
  Float(f64),
  Int(i64),
  None
}

impl From<i64> for ConstNumber {
  fn from(v: i64) -> ConstNumber { ConstNumber::Int(v) }
}

impl From<f64> for ConstNumber {
  fn from(v: f64) -> ConstNumber { ConstNumber::Float(v) }
}

impl ConstNumber {
  pub fn to_value<C: Custom>(&self) -> Option<Value<C>> {
    match self {
      Self::Float(f) => Some(Value::Float(*f)),
      Self::Int(i) => Some(Value::Int(*i)),
      Self::None => None
    }
  }
}

/// An array type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Array<C: Custom> {
  types: Vec<Type<C>>
}

impl<C: Custom> Array<C> {
  pub fn new(types: Vec<Type<C>>) -> Array<C> {
    assert!(types.iter().map(|t| t.flat_len()).sum::<usize>() <= MAX_CONST_ELEMENTS);
    Array { types }
  }

  /// The length of this array, if it was flattened. This includes the individual lengths of nested arrays and
  /// objects.
  pub fn flat_len(&self) -> usize { self.types.iter().map(|t| t.flat_len()).sum() }

  pub fn types(&self) -> &[Type<C>] { &self.types }
  pub fn into_types(self) -> Vec<Type<C>> { self.types }
  pub fn get(&self, ind: usize) -> &Type<C> { self.types.get(ind).unwrap() }
  pub fn get_mut(&mut self, ind: usize) -> &mut Type<C> { self.types.get_mut(ind).unwrap() }
  pub fn len(&self) -> usize { self.types.len() }
  pub fn is_empty(&self) -> bool { self.types.is_empty() }
  pub fn add(&mut self, t: Type<C>) { self.types.push(t); }
  pub fn is_single_use(&self) -> bool { self.types.iter().any(|v| v.is_single_use()) }

  pub fn similar(&self, other: &Self) -> bool {
    self.types.len() == other.types.len() && self.types.iter().zip(other.types.iter()).all(|(t1, t2)| t1.similar(t2))
  }
}

/// An object type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Object<C: Custom> {
  array: Vec<(String, Type<C>)>
}

impl<C: Custom> Object<C> {
  pub fn new(array: Vec<(String, Type<C>)>) -> Object<C> {
    assert!(is_sorted_by(&array, |a| a[0].0 < a[1].0));
    Object { array }
  }

  /// The length of this object, if it was flattened. This includes the individual lengths of nested arrays and
  /// objects.
  pub fn flat_len(&self) -> usize { self.array.iter().map(|(_, t)| t.flat_len()).sum() }

  pub fn array(&self) -> &[(String, Type<C>)] { &self.array }
  pub fn index_of(&self, key: &str) -> Option<usize> { self.array.iter().position(|(k, _)| k == key) }
  pub fn contains_keys(&self, keys: &[&str]) -> bool { keys.iter().all(|key| self.array.iter().any(|(k, _)| k == key)) }
  pub fn get(&self, key: &str) -> &Type<C> { &self.array.iter().find(|(k, _)| k == key).unwrap().1 }
  pub fn at(&self, ind: usize) -> &Type<C> { &self.array.get(ind).unwrap().1 }
  pub fn at_mut(&mut self, ind: usize) -> &mut Type<C> { &mut self.array.get_mut(ind).unwrap().1 }
  pub fn len(&self) -> usize { self.array.len() }
  pub fn is_empty(&self) -> bool { self.array.is_empty() }
  pub fn is_single_use(&self) -> bool { self.array.iter().any(|(_, t)| t.is_single_use()) }
  pub fn ordered(&self) -> impl Iterator<Item = &Type<C>> + '_ { self.array.iter().map(|(_, t)| t) }
  pub fn into_ordered(self) -> impl Iterator<Item = Type<C>> { self.array.into_iter().map(|(_, t)| t) }

  pub fn similar(&self, other: &Self) -> bool {
    self.array.len() == other.array.len()
      && self.array.iter().zip(other.array.iter()).all(|((k1, t1), (k2, t2))| k1 == k2 && t1.similar(t2))
  }

  pub fn add(&mut self, key: String, t: Type<C>) {
    // sort the keys lexically, to keep every equivalent type Eq
    self.array.push((key, t));
    self.array.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
  }
}

fn is_sorted_by<T, F: FnMut(&[T]) -> bool>(data: &[T], f: F) -> bool { data.windows(2).all(f) }
