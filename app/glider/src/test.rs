//! A module to help with testing.

use std::sync::Once;

static INIT: Once = Once::new();

pub fn setup() {
  use tracing_subscriber::{fmt, prelude::*, EnvFilter};
  INIT.call_once(|| tracing_subscriber::registry().with(fmt::layer()).with(EnvFilter::from_default_env()).init());
}
