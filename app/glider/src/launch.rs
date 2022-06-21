//! The launch point for glider; this module sets up logging, interrupts, the asynchronous runtime, and the
//! top-level error handling.

#![recursion_limit = "1024"]

mod main;

use std::process;
use tokio::runtime::Runtime;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn main() {
  let format = fmt::format()
    .with_level(true)
    .with_target(false)
    .with_thread_ids(false)
    .with_thread_names(false)
    .with_source_location(false)
    .pretty()
    .with_source_location(false);

  // TODO(later): use production-ready formatting.
  tracing_subscriber::registry().with(fmt::layer().event_format(format)).with(EnvFilter::from_default_env()).init();

  ctrlc::set_handler(graceful_exit_sigint).expect("Can't install interrupt handler.");

  if let Err(e) = Runtime::new().unwrap().block_on(crate::main::main()) {
    eprintln!("Error: {}", e);
    e.chain().skip(1).for_each(|cause| eprintln!("Caused by: {}", cause));

    // // Try running with `RUST_BACKTRACE=1` for a backtrace
    // if let Some(backtrace) = e.backtrace() {
    //   writeln!(stderr, "Backtrace:\n{:?}", backtrace).expect(errmsg);
    // }

    process::exit(1);
  }
}

#[cfg(not(test))]
pub fn graceful_exit_sigint() {
  use tracing::warn;

  warn!("Received SIGINT: exiting");
  std::process::exit(0);
}

#[cfg(test)]
pub fn graceful_exit_sigint() {}
