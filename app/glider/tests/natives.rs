//! A set of native function tests for the alchemy vm.
//!
//! The tests in this suite use some native functions, defined in `util`:
//! - `fourty_two()`: returns the number 42.
//! - `get_name(obj)`: returns the obj.name string.
//! - `recall(f)`: calls the given function.
//! - `reflex(v)`: returns a no-arg function that returns the given value.

mod util;

use glider::test::setup;
use util::{expect_i64, expect_str};

#[tokio::test]
async fn fourty_two() {
  setup();
  expect_i64("fourty_two()", 42).await;
}

#[tokio::test]
async fn extract_obj() {
  setup();
  expect_str(r#"get_name({ name: "bob" })"#, "bob").await;
}

#[tokio::test]
async fn native_to_alc() {
  setup();
  expect_i64("f=fn() { 1 }; recall(f) + 2", 3).await;
}

#[tokio::test]
async fn alc_to_native() {
  setup();
  expect_i64("f=fn() { fourty_two() }; f() + 1", 43).await;
}

#[tokio::test]
async fn return_fn() {
  setup();
  expect_i64("reflex(4)()", 4).await;
}

// TODO(HERE): Test a `recall`-like native that attempts to compile a given funtion multiple times, to verify
// that the status checks in compiler::pass_1 and compiler::pass_2 are also available to native functions.
// Otherwise if a native attempts to compile a function that is already completed, the assert in
// callable::pending_for will trip.
