//! Test utility.

use glider::callable::{BuildData, NativeBuildFn, NativeDef, NativeRunFn, ObjMap, RunData};
use glider::compiler::{build, compile_with_natives, Pass};
use glider::custom::Custom;
use glider::enhancer::enhance;
use glider::parser::parse;
use glider::types::{ConstNumber, Type};
use glider::value::Value;
use glider::vm::run;
use glider_macros::{build_fn, run_fn};

pub async fn run_script(content: &str) -> Value<()> {
  let script = parse(content).await;
  let enhanced = enhance(script);
  let (fn_def, ntv_vals) = compile_with_natives(enhanced, natives()).await;
  glider::vm::run_script(fn_def.inner(), ntv_vals).await
}

#[allow(dead_code)]
pub async fn expect_i64(content: &str, expected: i64) {
  assert_eq!(run_script(content).await.as_int(), expected);
}

#[allow(dead_code)]
pub async fn expect_str(content: &str, expected: &str) {
  assert_eq!(run_script(content).await.as_str(), expected);
}

fn natives<'a, C: Custom>() -> &'a [(NativeBuildFn<C>, NativeRunFn<C>, &'a str)] {
  &[
    (fourty_two_build_fn, fourty_two_run_fn, "fourty_two"),
    (get_name_build_fn, get_name_run_fn, "get_name"),
    (recall_build_fn, recall_run_fn, "recall"),
    (reflex_build_fn, reflex_run_fn, "reflex")
  ]
}

#[build_fn]
async fn fourty_two_build_fn<C: Custom>(
  _args: Vec<Type<C>>, _capts: Vec<Type<C>>, _pass: &mut Pass<C>
) -> (Type<C>, BuildData<C>) {
  (Type::Number(ConstNumber::None), BuildData::new())
}

#[run_fn]
async fn fourty_two_run_fn<C: Custom>(
  _args: Vec<Value<C>>, _capts: Vec<Value<C>>, _build: BuildData<C>, _run: RunData<C>
) -> Value<C> {
  Value::Int(42)
}

#[build_fn]
async fn get_name_build_fn<C: Custom>(
  args: Vec<Type<C>>, _capts: Vec<Type<C>>, _pass: &mut Pass<C>
) -> (Type<C>, BuildData<C>) {
  let mut build_data = BuildData::new();
  build_data.push_obj_map(ObjMap::for_object(args[0].as_object(), &["name"]));
  (Type::String(None), build_data)
}

#[run_fn]
async fn get_name_run_fn<C: Custom>(
  mut args: Vec<Value<C>>, _capts: Vec<Value<C>>, build: BuildData<C>, _run: RunData<C>
) -> Value<C> {
  build.obj_maps()[0].read(args[0].as_array_mut(), 0).shift()
}

#[build_fn]
async fn recall_build_fn<C: Custom>(
  args: Vec<Type<C>>, _capts: Vec<Type<C>>, pass: &mut Pass<C>
) -> (Type<C>, BuildData<C>) {
  let (fn_ind, fn_type) = build(&args[0], Vec::new(), pass).await;
  (fn_type, BuildData::new().with_fn_ind(fn_ind))
}

#[run_fn]
async fn recall_run_fn<C: Custom>(
  args: Vec<Value<C>>, _capts: Vec<Value<C>>, build: BuildData<C>, run_data: RunData<C>
) -> Value<C> {
  run(
    args.into_iter().next().unwrap(),
    build.into_fn_inds().into_iter().next().unwrap(),
    Vec::new(),
    run_data.status().clone()
  )
  .await
}

#[build_fn]
async fn reflex_build_fn<C: Custom>(
  args: Vec<Type<C>>, _capts: Vec<Type<C>>, pass: &mut Pass<C>
) -> (Type<C>, BuildData<C>) {
  // TODO(later): this is super clunky, and depends on `define_native` always getting called in the same order
  // on every pass. Figure out a better way.
  let gen_def = pass.define_native(|| {
    let mut gen_def = NativeDef::new("reflex", reflex_ret_build, reflex_ret_run);
    gen_def.set_captures(vec![("arg1".into(), args.into_iter().next().unwrap())]);
    gen_def
  });

  let mut build_data = BuildData::new();
  build_data.push_native_gen(gen_def.inner());

  (Type::Native(gen_def), build_data)
}

#[run_fn]
async fn reflex_run_fn<C: Custom>(
  args: Vec<Value<C>>, _capts: Vec<Value<C>>, build: BuildData<C>, _run: RunData<C>
) -> Value<C> {
  let gen_def = build.native_gens()[0].clone();
  Value::NativeDef(gen_def, args, Default::default())
}

#[build_fn]
async fn reflex_ret_build<C: Custom>(
  _args: Vec<Type<C>>, capts: Vec<Type<C>>, _pass: &mut Pass<C>
) -> (Type<C>, BuildData<C>) {
  (capts.into_iter().next().unwrap(), BuildData::new())
}

#[run_fn]
async fn reflex_ret_run<C: Custom>(
  _args: Vec<Value<C>>, capts: Vec<Value<C>>, _build: BuildData<C>, _run: RunData<C>
) -> Value<C> {
  capts.into_iter().next().unwrap()
}
