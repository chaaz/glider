//! A set of closure tests for the alchemy vm.

mod util;

use glider::test::setup;
use util::expect_i64;

#[tokio::test]
async fn unique_create() {
  setup();

  expect_i64(
    r#"
f = fn(a) { fn() { a } };
three_f = f(3);
four_f = f(4);
three_f() + four_f()
    "#,
    7
  )
  .await;
}
