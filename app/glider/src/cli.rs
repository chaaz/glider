//! The command-line options for Glider.

use crate::callable::{BuildData, RunData};
use crate::compiler::{compile_with_natives, Pass};
use crate::enhancer::enhance;
use crate::errors::Result;
use crate::parser::parse;
use crate::types::{ConstNumber, Type};
use crate::value::Value;
use crate::vm::run_script;
use clap::Parser;
use glider_macros::{build_fn, run_fn};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
  #[clap(short, long)]
  script: String
}

/// Get the values from the expected command-line options.
pub async fn execute() -> Result<()> {
  let args = Args::parse();
  run_file(&args.script).await
}

async fn run_file(input: &str) -> Result<()> {
  let content = std::fs::read_to_string(input)?;

  let script = parse(&content).await;
  let enhanced = enhance(script);
  let (fn_def, ntv_vals) = compile_with_natives(enhanced, &[(clock_build_fn, clock_run_fn, "clock")]).await;
  let value = run_script(fn_def.inner(), ntv_vals).await;

  println!("Done: {:?}", value);
  Ok(())
}

#[build_fn]
async fn clock_build_fn(
  _args: Vec<Type<()>>, _capts: Vec<Type<()>>, _pass: &mut Pass<()>
) -> (Type<()>, BuildData<()>) {
  (Type::Number(ConstNumber::None), BuildData::new())
}

#[run_fn]
async fn clock_run_fn(
  _args: Vec<Value<()>>, _capts: Vec<Value<()>>, _build: BuildData<()>, _run: RunData<()>
) -> Value<()> {
  Value::Int(1)
}
