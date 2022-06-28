//! The customization parameters for glider, which lets clients specify new types, values, and runtime status.

use crate::errors::Result;
use std::fmt;
use std::hash::Hash;

pub trait Custom: fmt::Debug + Clone + Default + PartialEq + Eq + std::hash::Hash + Send + 'static {
  type Value: CustomValue<Self>;
  type RepeatValue: CustomRepeatValue<Self>;
  type Type: CustomType;
  type Status: CustomStatus;
  type Capture: CustomCapture;
}

impl Custom for () {
  type Value = ();
  type RepeatValue = ();
  type Type = ();
  type Status = ();
  type Capture = ();
}

pub trait CustomValue<C: Custom>: Sized + fmt::Debug + Send + 'static {
  fn shift(&mut self) -> Option<Self>;
  fn try_clone(&self) -> Result<Self>;
  fn repeatable(self) -> C::RepeatValue;
}

impl CustomValue<()> for () {
  fn shift(&mut self) -> Option<()> { Some(()) }
  fn try_clone(&self) -> Result<()> { Ok(()) }
  fn repeatable(self) {}
}

pub trait CustomRepeatValue<C: Custom>: Sized + fmt::Debug + Clone + Send + Sync + 'static {
  fn into_value(self) -> C::Value;
}

impl CustomRepeatValue<()> for () {
  fn into_value(self) {}
}

pub trait CustomType: fmt::Debug + Hash + PartialEq + Eq + Clone + Send + Sync + 'static {
  fn similar(&self, other: &Self) -> bool;
}

impl CustomType for () {
  fn similar(&self, _other: &Self) -> bool { true }
}

pub trait CustomStatus: Default + fmt::Debug + Clone + Send + Sync + 'static {}

impl CustomStatus for () {}

pub trait CustomCapture: Default + fmt::Debug + Clone + Send + Sync + 'static {}

impl CustomCapture for () {}
