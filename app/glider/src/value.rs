//! The values representable in our language.

use crate::callable::{FnDefInner, NativeDefInner};
use crate::custom::{Custom, CustomValue};
use crate::errors::Result;
use crate::pick;
use serde_json::{Number, Value as Json};
use std::fmt;
use std::mem;
use std::sync::{Arc, Mutex};

/// A values as they exist in the VM at runtime.
pub enum Value<C: Custom> {
  /// A unit placeholder, returned by empty blocks.
  Unit,

  Float(f64),
  Int(i64),
  Bool(bool),
  String(Arc<str>),
  Array(Vec<Value<C>>),
  Json(Json),
  FnDef(Arc<Mutex<FnDefInner<C>>>, Vec<Value<C>>),
  NativeDef(Arc<Mutex<NativeDefInner<C>>>, Vec<Value<C>>, C::Capture),
  Custom(C::Value),

  /// An evacuated value, or part of a constexpr array whose value is unknown.
  Void
}

impl<C: Custom> fmt::Display for Value<C> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Unit => write!(f, "()"),
      Self::Float(v) => write!(f, "{}", v),
      Self::Int(v) => write!(f, "{}", v),
      Self::Bool(v) => write!(f, "{}", v),
      Self::String(v) => write!(f, "{}", v),
      Self::Array(v) => {
        write!(f, "[")?;
        for (i, v) in v.iter().enumerate() {
          if i != 0 {
            write!(f, ", ")?;
          }
          v.fmt(f)?;
        }
        write!(f, "]")
      }
      Self::Json(v) => write!(f, "{}", v),
      Self::FnDef(_, _) => write!(f, "<function>"),
      Self::NativeDef(..) => write!(f, "<native>"),
      Self::Custom(_) => write!(f, "_custom value_"),
      Self::Void => write!(f, "-")
    }
  }
}

impl<C: Custom> fmt::Debug for Value<C> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Unit => write!(f, "()"),
      Self::Float(v) => write!(f, "{}", v),
      Self::Int(v) => write!(f, "{}", v),
      Self::Bool(v) => write!(f, "{}", v),
      Self::String(v) => write!(f, "\"{}\"", v),
      Self::Array(v) => write!(f, "{:?}", v),
      Self::Json(v) => write!(f, "{}", v),
      Self::FnDef(v, _) => v.try_lock().unwrap().fmt(f),
      Self::NativeDef(v, ..) => v.try_lock().unwrap().fmt(f),
      Self::Custom(_) => write!(f, "_custom value_"),
      Self::Void => write!(f, "-")
    }
  }
}

impl<C: Custom> From<i64> for Value<C> {
  fn from(v: i64) -> Value<C> { Value::Int(v) }
}

impl<C: Custom> From<f64> for Value<C> {
  fn from(v: f64) -> Value<C> { Value::Float(v) }
}

impl<C: Custom> From<bool> for Value<C> {
  fn from(v: bool) -> Value<C> { Value::Bool(v) }
}

impl<C: Custom> From<String> for Value<C> {
  fn from(v: String) -> Value<C> { Value::String(v.into()) }
}

impl<C: Custom> From<&str> for Value<C> {
  fn from(v: &str) -> Value<C> { v.to_string().into() }
}

impl<C: Custom> From<Json> for Value<C> {
  fn from(v: Json) -> Value<C> { Value::Json(v) }
}

impl<C: Custom> Value<C> {
  pub fn is_unit(&self) -> bool { matches!(self, Self::Unit) }
  pub fn is_bool(&self) -> bool { matches!(self, Self::Bool(_)) }
  pub fn is_int(&self) -> bool { matches!(self, Self::Int(_)) }
  pub fn is_float(&self) -> bool { matches!(self, Self::Float(_)) }
  pub fn is_number(&self) -> bool { matches!(self, Self::Float(_)) }
  pub fn is_json(&self) -> bool { matches!(self, Self::Json(_)) }
  pub fn is_string(&self) -> bool { matches!(self, Self::String(_)) }
  pub fn is_array(&self) -> bool { matches!(self, Self::Array(_)) }
  pub fn is_fn(&self) -> bool { matches!(self, Self::FnDef(..)) }
  pub fn is_native(&self) -> bool { matches!(self, Self::NativeDef(..)) }
  pub fn is_callable(&self) -> bool { matches!(self, Self::FnDef(..) | Self::NativeDef(..)) }
  pub fn is_custom(&self) -> bool { matches!(self, Self::Custom(_)) }

  pub fn as_bool(&self) -> bool { pick!(self, Self::Bool(v) => *v, "Not a boolean: {:?}") }
  pub fn as_int(&self) -> i64 { pick!(self, Self::Int(v) => *v, "Not an int: {:?}") }
  pub fn as_float(&self) -> f64 { pick!(self, Self::Float(v) => *v, "Not a float: {:?}") }
  pub fn as_str(&self) -> &str { pick!(self, Self::String(v) => v, "Not a string: {:?}") }
  pub fn as_array(&self) -> &[Value<C>] { pick!(self, Self::Array(v) => v, "Not an array: {:?}") }
  pub fn as_array_mut(&mut self) -> &mut Vec<Value<C>> { pick!(self, Self::Array(v) => v, "Not an array: {:?}") }
  pub fn as_json(&self) -> &Json { pick!(self, Self::Json(v) => v, "Not JSON: {:?}") }
  pub fn as_json_mut(&mut self) -> &mut Json { pick!(self, Self::Json(v) => v, "Not json: {:?}") }
  pub fn as_custom(&self) -> &C::Value { pick!(self, Self::Custom(v) => v, "Not a custom value: {:?}") }

  pub fn as_fn(&self) -> (Arc<Mutex<FnDefInner<C>>>, &[Value<C>]) {
    pick!(self, Self::FnDef(v, c) => (v.clone(), c), "Not a function: {:?}")
  }

  pub fn as_fn_mut(&mut self) -> (Arc<Mutex<FnDefInner<C>>>, &mut Vec<Value<C>>) {
    pick!(self, Self::FnDef(v, c) => (v.clone(), c), "Not a function: {:?}")
  }

  pub fn into_fn(self) -> (Arc<Mutex<FnDefInner<C>>>, Vec<Value<C>>) {
    pick!(self, Self::FnDef(v, c) => (v, c), "Not a function: {:?}")
  }

  pub fn as_native(&self) -> Arc<Mutex<NativeDefInner<C>>> {
    pick!(self, Self::NativeDef(v, ..) => v.clone(), "Not native: {:?}")
  }

  #[allow(clippy::type_complexity)]
  pub fn as_native_mut(&mut self) -> (Arc<Mutex<NativeDefInner<C>>>, &mut Vec<Value<C>>, &mut C::Capture) {
    pick!(self, Self::NativeDef(v, a, c) => (v.clone(), a, c), "Not native: {:?}")
  }

  pub fn into_json(self) -> Json { pick!(self, Self::Json(v) => v, "Not JSON: {:?}") }
  pub fn into_string(self) -> String { pick!(self, Self::String(v) => v.to_string(), "Not a string: {:?}") }
  pub fn into_array(self) -> Vec<Value<C>> { pick!(self, Self::Array(v) => v, "Not an array: {:?}") }
  pub fn into_custom(self) -> C::Value { pick!(self, Self::Custom(v) => v, "Not a custom value: {:?}") }

  /// Act like `clone` for most value types; but for streaming or otherwise un-cloneable values, take the value
  /// instead, leaving a `Void` in its place.
  pub fn shift(&mut self) -> Value<C> {
    match self {
      Self::Unit => Self::Unit,
      Self::Float(v) => Self::Float(*v),
      Self::Int(v) => Self::Int(*v),
      Self::Bool(v) => Self::Bool(*v),
      Self::String(v) => Self::String(v.clone()),
      Self::Array(v) => Self::Array(v.iter_mut().map(|v| v.shift()).collect()),
      Self::Json(v) => Self::Json(v.clone()),
      Self::FnDef(v, c) => Self::FnDef(v.clone(), c.iter_mut().map(|v| v.shift()).collect()),
      Self::NativeDef(v, c, cc) => Self::NativeDef(v.clone(), c.iter_mut().map(|v| v.shift()).collect(), cc.clone()),
      Self::Custom(v) => match v.shift() {
        Some(v) => Self::Custom(v),
        None => Self::Custom(mem::replace(self, Self::Void).into_custom())
      },
      Self::Void => panic!("Cannot shift a voided value.")
    }
  }

  pub fn try_clone(&self) -> Result<Value<C>> {
    match self {
      Self::Unit => Ok(Self::Unit),
      Self::Float(v) => Ok(Self::Float(*v)),
      Self::Int(v) => Ok(Self::Int(*v)),
      Self::Bool(v) => Ok(Self::Bool(*v)),
      Self::String(v) => Ok(Self::String(v.clone())),
      Self::Array(v) => Ok(Self::Array(v.iter().map(|v| v.try_clone()).collect::<Result<_>>()?)),
      Self::Json(v) => Ok(Self::Json(v.clone())),
      Self::FnDef(v, c) => Ok(Self::FnDef(v.clone(), c.iter().map(|v| v.try_clone()).collect::<Result<_>>()?)),
      Self::NativeDef(v, c, cc) => {
        Ok(Self::NativeDef(v.clone(), c.iter().map(|v| v.try_clone()).collect::<Result<_>>()?, cc.clone()))
      }
      Self::Custom(v) => v.try_clone().map(Self::Custom),
      Self::Void => panic!("Cannot clone a voided value.")
    }
  }

  pub fn op_negate(&self) -> Value<C> {
    match self {
      Self::Int(v) => Self::Int(-*v),
      Self::Float(v) => Self::Float(-*v),
      Self::Json(Json::Number(n)) if n.is_u64() => Self::Json(Json::Number((-(n.as_u64().unwrap() as i64)).into())),
      Self::Json(Json::Number(n)) if n.is_i64() => Self::Json(Json::Number((-n.as_i64().unwrap()).into())),
      Self::Json(Json::Number(n)) if n.is_f64() => {
        Self::Json(Json::Number(Number::from_f64(-n.as_f64().unwrap()).unwrap()))
      }
      _ => panic!("No negation for {:?}", self)
    }
  }

  pub fn op_not(&self) -> Value<C> {
    match self {
      Self::Bool(v) => Self::Bool(!*v),
      Self::Json(Json::Bool(n)) => Self::Bool(!n),
      _ => panic!("No negation for {:?}", self)
    }
  }

  pub fn op_add(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Float(v1 + v2),
      (Self::Int(v1), Value::Int(v2)) => Self::Int(v1 + v2),
      (Self::String(v1), Value::String(v2)) => Self::String(concat(v1, v2).into()),
      (Self::Int(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 + b.as_f64().unwrap()).unwrap()))
      }
      (Self::Float(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 + b.as_f64().unwrap()).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Int(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() + *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Float(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() + *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() + b.as_f64().unwrap()).unwrap()))
      }
      (Self::String(a), Self::Json(Json::String(b))) => Self::Json(Json::String(concat(a, b))),
      (Self::Json(Json::String(a)), Self::String(b)) => Self::Json(Json::String(concat(a, b))),
      (Self::Json(Json::String(a)), Self::Json(Json::String(b))) => Self::Json(Json::String(concat(a, b))),
      (o1, o2) => panic!("Can't add mismatch types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_subtract(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Float(v1 - v2),
      (Self::Int(v1), Value::Int(v2)) => Self::Int(v1 - v2),
      (Self::Int(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 - b.as_f64().unwrap()).unwrap()))
      }
      (Self::Float(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 - b.as_f64().unwrap()).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Int(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() - *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Float(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() - *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() - b.as_f64().unwrap()).unwrap()))
      }
      (o1, o2) => panic!("Can't subtract mismatch types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_multiply(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Float(v1 * v2),
      (Self::Int(v1), Value::Int(v2)) => Self::Int(v1 * v2),
      (Self::Int(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 * b.as_f64().unwrap()).unwrap()))
      }
      (Self::Float(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 * b.as_f64().unwrap()).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Int(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() * *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Float(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() * *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() * b.as_f64().unwrap()).unwrap()))
      }
      (o1, o2) => panic!("Can't multiply mismatch types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_divide(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Float(v1 / v2),
      (Self::Int(v1), Value::Int(v2)) => Self::Int(v1 / v2),
      (Self::Int(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 / b.as_f64().unwrap()).unwrap()))
      }
      (Self::Float(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 / b.as_f64().unwrap()).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Int(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() / *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Float(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() / *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() / b.as_f64().unwrap()).unwrap()))
      }
      (o1, o2) => panic!("Can't divide mismatch types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_mod(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Float(v1 % v2),
      (Self::Int(v1), Value::Int(v2)) => Self::Int(v1 % v2),
      (Self::Int(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 % b.as_f64().unwrap()).unwrap()))
      }
      (Self::Float(a), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(*a as f64 % b.as_f64().unwrap()).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Int(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() % *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Float(b)) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() % *b as f64).unwrap()))
      }
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Json(Json::Number(Number::from_f64(a.as_f64().unwrap() % b.as_f64().unwrap()).unwrap()))
      }
      (o1, o2) => panic!("Can't divide mismatch types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_gt(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Bool(v1 > v2),
      (Self::Int(v1), Value::Int(v2)) => Self::Bool(v1 > v2),
      (Self::Int(a), Self::Json(Json::Number(b))) => Self::Bool(*a as f64 > b.as_f64().unwrap()),
      (Self::Float(a), Self::Json(Json::Number(b))) => Self::Bool(*a as f64 > b.as_f64().unwrap()),
      (Self::Json(Json::Number(a)), Self::Int(b)) => Self::Bool(a.as_f64().unwrap() > *b as f64),
      (Self::Json(Json::Number(a)), Self::Float(b)) => Self::Bool(a.as_f64().unwrap() > *b as f64),
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Bool(a.as_f64().unwrap() > b.as_f64().unwrap())
      }
      (o1, o2) => panic!("Can't compare types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_gte(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Bool(v1 >= v2),
      (Self::Int(v1), Value::Int(v2)) => Self::Bool(v1 >= v2),
      (Self::Int(a), Self::Json(Json::Number(b))) => Self::Bool(*a as f64 >= b.as_f64().unwrap()),
      (Self::Float(a), Self::Json(Json::Number(b))) => Self::Bool(*a as f64 >= b.as_f64().unwrap()),
      (Self::Json(Json::Number(a)), Self::Int(b)) => Self::Bool(a.as_f64().unwrap() >= *b as f64),
      (Self::Json(Json::Number(a)), Self::Float(b)) => Self::Bool(a.as_f64().unwrap() >= *b as f64),
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Bool(a.as_f64().unwrap() >= b.as_f64().unwrap())
      }
      (o1, o2) => panic!("Can't compare types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_lt(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Bool(v1 < v2),
      (Self::Int(v1), Value::Int(v2)) => Self::Bool(v1 < v2),
      (Self::Int(a), Self::Json(Json::Number(b))) => Self::Bool((*a as f64) < b.as_f64().unwrap()),
      (Self::Float(a), Self::Json(Json::Number(b))) => Self::Bool((*a as f64) < b.as_f64().unwrap()),
      (Self::Json(Json::Number(a)), Self::Int(b)) => Self::Bool(a.as_f64().unwrap() < *b as f64),
      (Self::Json(Json::Number(a)), Self::Float(b)) => Self::Bool(a.as_f64().unwrap() < *b as f64),
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Bool(a.as_f64().unwrap() < b.as_f64().unwrap())
      }
      (o1, o2) => panic!("Can't compare types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_lte(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Bool(v1 <= v2),
      (Self::Int(v1), Value::Int(v2)) => Self::Bool(v1 <= v2),
      (Self::Int(a), Self::Json(Json::Number(b))) => Self::Bool(*a as f64 <= b.as_f64().unwrap()),
      (Self::Float(a), Self::Json(Json::Number(b))) => Self::Bool(*a as f64 <= b.as_f64().unwrap()),
      (Self::Json(Json::Number(a)), Self::Int(b)) => Self::Bool(a.as_f64().unwrap() <= *b as f64),
      (Self::Json(Json::Number(a)), Self::Float(b)) => Self::Bool(a.as_f64().unwrap() <= *b as f64),
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Bool(a.as_f64().unwrap() <= b.as_f64().unwrap())
      }
      (o1, o2) => panic!("Can't compare types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_eq(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Float(v1), Value::Float(v2)) => Self::Bool(is_eq(*v1, *v2)),
      (Self::Int(v1), Value::Int(v2)) => Self::Bool(v1 == v2),
      (Self::Int(a), Self::Json(Json::Number(b))) => Self::Bool(is_eq(*a as f64, b.as_f64().unwrap())),
      (Self::Float(a), Self::Json(Json::Number(b))) => Self::Bool(is_eq(*a as f64, b.as_f64().unwrap())),
      (Self::Json(Json::Number(a)), Self::Int(b)) => Self::Bool(is_eq(a.as_f64().unwrap(), *b as f64)),
      (Self::Json(Json::Number(a)), Self::Float(b)) => Self::Bool(is_eq(a.as_f64().unwrap(), *b as f64)),
      (Self::Json(Json::Number(a)), Self::Json(Json::Number(b))) => {
        Self::Bool(is_eq(a.as_f64().unwrap(), b.as_f64().unwrap()))
      }
      (Self::String(v1), Value::String(v2)) => Self::Bool(v1 == v2),
      (Self::Json(Json::String(a)), Self::String(b)) => Self::Bool(a.as_str() == b.as_ref()),
      (Self::String(a), Self::Json(Json::String(b))) => Self::Bool(a.as_ref() == b.as_str()),
      (Self::Bool(v1), Value::Bool(v2)) => Self::Bool(v1 == v2),
      (Self::Bool(a), Self::Json(Json::Bool(b))) => Self::Bool(a == b),
      (Self::Json(Json::Bool(a)), Self::Bool(b)) => Self::Bool(a == b),
      (Self::FnDef(a, _), Self::FnDef(b, _)) => Self::Bool(Arc::ptr_eq(a, b)), // TODO: capture eq
      (Self::NativeDef(a, ..), Self::NativeDef(b, ..)) => Self::Bool(Arc::ptr_eq(a, b)),
      (o1, o2) => panic!("Can't compare types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_neq(&self, other: &Value<C>) -> Value<C> { Value::Bool(!self.op_eq(other).as_bool()) }

  pub fn op_and(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Bool(v1), Value::Bool(v2)) => Self::Bool(*v1 && *v2),
      (Self::Bool(v1), Value::Json(Json::Bool(v2))) => Self::Bool(*v1 && *v2),
      (Value::Json(Json::Bool(v1)), Self::Bool(v2)) => Self::Bool(*v1 && *v2),
      (Value::Json(Json::Bool(v1)), Self::Json(Json::Bool(v2))) => Self::Bool(*v1 && *v2),
      (o1, o2) => panic!("Can't compare types: {:?}, {:?}", o1, o2)
    }
  }

  pub fn op_or(&self, other: &Value<C>) -> Value<C> {
    match (self, other) {
      (Self::Bool(v1), Value::Bool(v2)) => Self::Bool(*v1 || *v2),
      (Self::Bool(v1), Value::Json(Json::Bool(v2))) => Self::Bool(*v1 || *v2),
      (Value::Json(Json::Bool(v1)), Self::Bool(v2)) => Self::Bool(*v1 || *v2),
      (Value::Json(Json::Bool(v1)), Self::Json(Json::Bool(v2))) => Self::Bool(*v1 || *v2),
      (o1, o2) => panic!("Can't compare types: {:?}, {:?}", o1, o2)
    }
  }
}

fn concat(v1: &impl ToString, v2: &impl ToString) -> String { format!("{}{}", v1.to_string(), v2.to_string()) }

fn is_eq(v1: f64, v2: f64) -> bool { (v1 - v2).abs() < f64::EPSILON }

/// Atomic literal values that exist in the source code.
#[derive(Clone, Debug)]
pub enum Literal {
  Unit,
  Float(f64),
  Int(i64),
  Bool(bool),
  String(Arc<str>)
}

impl From<i64> for Literal {
  fn from(v: i64) -> Literal { Literal::Int(v) }
}

impl From<f64> for Literal {
  fn from(v: f64) -> Literal { Literal::Float(v) }
}

impl From<bool> for Literal {
  fn from(v: bool) -> Literal { Literal::Bool(v) }
}

impl From<String> for Literal {
  fn from(v: String) -> Literal { Literal::String(v.into()) }
}

impl From<&str> for Literal {
  fn from(v: &str) -> Literal { v.to_string().into() }
}

impl Literal {
  pub fn is_unit(&self) -> bool { matches!(self, Self::Unit) }
  pub fn is_bool(&self) -> bool { matches!(self, Self::Bool(_)) }
  pub fn is_int(&self) -> bool { matches!(self, Self::Int(_)) }
  pub fn is_float(&self) -> bool { matches!(self, Self::Float(_)) }
  pub fn is_number(&self) -> bool { matches!(self, Self::Float(_)) }
  pub fn is_string(&self) -> bool { matches!(self, Self::String(_)) }

  pub fn as_bool(&self) -> bool { pick!(self, Self::Bool(v) => *v, "Not a boolean: {:?}") }
  pub fn as_int(&self) -> i64 { pick!(self, Self::Int(v) => *v, "Not an int: {:?}") }
  pub fn as_float(&self) -> f64 { pick!(self, Self::Float(v) => *v, "Not a float: {:?}") }
  pub fn as_str(&self) -> &str { pick!(self, Self::String(v) => v, "Not a string: {:?}") }
}

/// Values that exist in the constants area of the compiled source code.
#[derive(Clone, Debug)]
pub enum Constant<C: Custom> {
  Unit,
  Float(f64),
  Int(i64),
  Bool(bool),
  String(Arc<str>),
  FnDef(Arc<Mutex<FnDefInner<C>>>),
  NativeDef(Arc<Mutex<NativeDefInner<C>>>)
}

impl<C: Custom> From<i64> for Constant<C> {
  fn from(v: i64) -> Constant<C> { Constant::Int(v) }
}

impl<C: Custom> From<f64> for Constant<C> {
  fn from(v: f64) -> Constant<C> { Constant::Float(v) }
}

impl<C: Custom> From<bool> for Constant<C> {
  fn from(v: bool) -> Constant<C> { Constant::Bool(v) }
}

impl<C: Custom> From<String> for Constant<C> {
  fn from(v: String) -> Constant<C> { Constant::String(v.into()) }
}

impl<C: Custom> From<&str> for Constant<C> {
  fn from(v: &str) -> Constant<C> { v.to_string().into() }
}

impl<C: Custom> From<Arc<Mutex<FnDefInner<C>>>> for Constant<C> {
  fn from(v: Arc<Mutex<FnDefInner<C>>>) -> Constant<C> { Constant::FnDef(v) }
}

impl<C: Custom> Constant<C> {
  pub fn is_unit(&self) -> bool { matches!(self, Self::Unit) }
  pub fn is_bool(&self) -> bool { matches!(self, Self::Bool(_)) }
  pub fn is_int(&self) -> bool { matches!(self, Self::Int(_)) }
  pub fn is_float(&self) -> bool { matches!(self, Self::Float(_)) }
  pub fn is_number(&self) -> bool { matches!(self, Self::Float(_)) }
  pub fn is_string(&self) -> bool { matches!(self, Self::String(_)) }
  pub fn is_fn(&self) -> bool { matches!(self, Self::FnDef(_)) }

  pub fn as_bool(&self) -> bool { pick!(self, Self::Bool(v) => *v, "Not a boolean: {:?}") }
  pub fn as_int(&self) -> i64 { pick!(self, Self::Int(v) => *v, "Not an int: {:?}") }
  pub fn as_float(&self) -> f64 { pick!(self, Self::Float(v) => *v, "Not a float: {:?}") }
  pub fn as_str(&self) -> &str { pick!(self, Self::String(v) => v, "Not a string: {:?}") }
  pub fn as_fn(&self) -> Arc<Mutex<FnDefInner<C>>> { pick!(self, Self::FnDef(v) => v.clone(), "Not a function: {:?}") }

  pub fn to_value(&self) -> Value<C> {
    match self {
      Self::Unit => Value::Unit,
      Self::Float(v) => Value::Float(*v),
      Self::Int(v) => Value::Int(*v),
      Self::Bool(v) => Value::Bool(*v),
      Self::String(v) => Value::String(v.to_string().into()),

      // Where this is used in Vm::closure, the new vector is then filled with the captured stack values.
      Self::FnDef(v) => Value::FnDef(v.clone(), Vec::new()),

      Self::NativeDef(v) => Value::NativeDef(v.clone(), Vec::new(), Default::default())
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn float_shift() {
    assert_eq!(Value::<()>::Float(1.0).shift().as_float(), 1.0);
  }
}
