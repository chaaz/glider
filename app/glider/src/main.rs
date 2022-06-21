//! The execution for Glider.

// This isn't actually the [[bin]] entrypoint for Glider: See `launch.rs` instead.

use glider::cli::execute;
use glider::errors::Result;

pub async fn main() -> Result<()> { execute().await }
