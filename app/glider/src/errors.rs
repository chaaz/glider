//! Error handling is based on the `anyhow` crate.

pub use anyhow::{Context, Result};
use std::fmt;

pub enum Error {
  Runtime(RuntimeError),
  Compile(CompileError)
}

impl Error {
  pub fn runtime(msg: &str) -> Error { Error::Runtime(msg.into()) }
  pub fn compile(msg: &str) -> Error { Error::Compile(msg.into()) }
}

impl std::error::Error for Error {}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Runtime(e) => e.fmt(f),
      Self::Compile(e) => e.fmt(f)
    }
  }
}

impl std::fmt::Debug for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::Runtime(e) => e.fmt(f),
      Self::Compile(e) => e.fmt(f)
    }
  }
}

pub struct RuntimeError {
  msg: String
}

impl From<&str> for RuntimeError {
  fn from(msg: &str) -> RuntimeError { RuntimeError { msg: msg.to_string() } }
}

impl fmt::Display for RuntimeError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.msg.fmt(f) }
}

impl fmt::Debug for RuntimeError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.msg.fmt(f) }
}

pub struct CompileError {
  msg: String
}

impl From<&str> for CompileError {
  fn from(msg: &str) -> CompileError { CompileError { msg: msg.to_string() } }
}

impl fmt::Display for CompileError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.msg.fmt(f) }
}

impl fmt::Debug for CompileError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.msg.fmt(f) }
}

#[macro_export]
macro_rules! rerr {
  ($($arg:tt)*) => (
    std::result::Result::Err($crate::errors::Error::Compile($crate::errors::CompileError { msg: format!($($arg)*) }))
  )
}

// #[macro_export]
// macro_rules! bail {
//   ($($arg:tt)*) => (return ($crate::err!($($arg)*)))
// }

// #[macro_export]
// macro_rules! pick {
//   ($s:expr, $( $p:pat => $e:expr ),*) => ( match $s { $( $p => $e ),* } );
// }

#[macro_export]
macro_rules! pick {
  ($s:expr, $p:pat => $e:expr, $er:expr) => {
    match $s {
      $p => $e,
      o => panic!($er, o)
    }
  };
}

#[macro_export]
macro_rules! pick_opt {
  ($s:expr, $p:pat => $e:expr) => {
    match $s {
      $p => std::option::Option::Some($e),
      _ => std::option::Option::None
    }
  };
}
