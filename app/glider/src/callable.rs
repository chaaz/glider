//! A set of structures that track callable values.
//!
//! There are two types of callable values: code functions (which are functions defined in glider source code),
//! and native functions (which are built and linked to the source at runtime). Compiling, building, and
//! tracking these functions during the build and run phases is largely the same process for both types, so the
//! structures here are generic around the `Callable` trait.
//!
//! The major implementation differences between code and native functions has to do with: argument storage
//! (code function arguments are destructured according to the function parameters before calling); compilation
//! (native functions don't need to generate bytecode); and invocation (code functions need to use a VM to
//! execute).

use crate::compiler::{find_extracts, Chunk, Compiler, Constants, Extract, FnInd, Opcode, Pass};
use crate::custom::Custom;
use crate::enhancer::Captured;
use crate::parser::{Block, Destructure, ExpressionMeta};
use crate::scanner::Position;
use crate::types::{Object, Type};
use crate::value::{Constant, Value};
use async_trait::async_trait;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};

/// A callable value, which is either a code function or native function.
#[async_trait]
pub trait Callable<C: Custom>: Clone + fmt::Debug {
  type Inner;
  type Impl: Default + fmt::Debug;

  fn extracts_and_args(def: &CallableDef<Self, C>, args: &[Type<C>]) -> (Vec<Extract>, Vec<Type<C>>);

  async fn unpack_args<'p>(
    this: &CallableDef<Self, C>, compiler: &mut Compiler<'p, C>, args: &[ExpressionMeta<Captured>]
  ) -> Vec<Type<C>>;

  async fn compile(
    this: &CallableDef<Self, C>, pending_impl: ImplRef<Self, C>, args: &[Type<C>], pass: &mut Pass<C>
  ) -> (Type<C>, Self::Impl);
}

/// A callable definition, which represents the abstract source execution. The definition also keeps a list
/// of specific implementations compiled for specific types arguments.
#[derive(Clone)]
pub struct CallableDef<T: Callable<C>, C: Custom> {
  def: T,
  inner: Arc<Mutex<CallableDefInner<T, C>>>
}

impl<T: Callable<C>, C: Custom> PartialEq for CallableDef<T, C> {
  fn eq(&self, other: &Self) -> bool { Arc::ptr_eq(&self.inner, &other.inner) }
}

impl<T: Callable<C>, C: Custom> Eq for CallableDef<T, C> {}

impl<T: Callable<C>, C: Custom> Hash for CallableDef<T, C> {
  fn hash<H: Hasher>(&self, state: &mut H) { state.write_usize(Arc::as_ptr(&self.inner) as usize); }
}

impl<T: Callable<C>, C: Custom> CallableDef<T, C> {
  pub fn def(&self) -> &T { &self.def }
  pub fn inner(&self) -> Arc<Mutex<CallableDefInner<T, C>>> { self.inner.clone() }

  pub async fn unpack_args<'p>(
    &self, compiler: &mut Compiler<'p, C>, args: &[ExpressionMeta<Captured>]
  ) -> Vec<Type<C>> {
    T::unpack_args(self, compiler, args).await
  }

  pub async fn compile(&self, args: &[Type<C>], pass: &mut Pass<C>) -> (usize, Type<C>) {
    let (morph_ind, pending_impl) = self.pending_for(args, pass);
    let (fn_type, data) = T::compile(self, pending_impl, args, pass).await;
    let status = pass.done_status(&fn_type);
    self.mark_done(args, status, data);
    (morph_ind, fn_type)
  }

  /// Return the pending-ness of the def, the morph_index (only valid if `status.is_completed()`), and the
  /// current impl status.
  pub fn find_status(&self, args: &[Type<C>]) -> (bool, (usize, ImplStatus<C>)) {
    let inner = self.inner.try_lock().unwrap();
    (inner.pending(), inner.status(args))
  }

  pub fn pending_for(&self, args: &[Type<C>], pass: &mut Pass<C>) -> (usize, ImplRef<T, C>) {
    self.inner.try_lock().unwrap().pending_for(args, pass)
  }

  pub fn mark_done(&self, args: &[Type<C>], status: ImplStatus<C>, data: T::Impl) {
    self.inner.try_lock().unwrap().mark_done(args, status, data)
  }
}

pub struct CallableDefInner<T: Callable<C>, C: Custom> {
  def: T::Inner,
  pending: bool,
  building: HashMap<Vec<Type<C>>, CallableImpl<T, C>>,
  #[allow(clippy::type_complexity)]
  completed: Vec<(Vec<Type<C>>, CallableImpl<T, C>)>
}

impl<T: Callable<C>, C: Custom> CallableDefInner<T, C> {
  pub fn new(def: T::Inner) -> CallableDefInner<T, C> {
    CallableDefInner { pending: false, building: HashMap::new(), completed: Vec::new(), def }
  }

  pub fn pending(&self) -> bool { self.pending }
  pub fn set_pending(&mut self, c: bool) { self.pending = c; }
  pub fn find_building(&self, args: &[Type<C>]) -> Option<&CallableImpl<T, C>> { self.building.get(args) }

  pub fn find_completed(&self, args: &[Type<C>]) -> Option<(usize, &CallableImpl<T, C>)> {
    self.completed.iter().enumerate().find(|(_, (a, _))| args == a).map(|(i, (_, impl_ref))| (i, impl_ref))
  }

  pub fn find(&self, args: &[Type<C>]) -> Option<(usize, &CallableImpl<T, C>)> {
    self.find_building(args).map(|b| (0, b)).or_else(|| self.find_completed(args))
  }

  pub fn find_building_mut(&mut self, args: &[Type<C>]) -> Option<&mut CallableImpl<T, C>> {
    self.building.get_mut(args)
  }

  pub fn find_completed_mut(&mut self, args: &[Type<C>]) -> Option<&mut CallableImpl<T, C>> {
    self.completed.iter_mut().find(|(a, _)| args == a).map(|(_, fn_impl)| fn_impl)
  }

  pub fn status(&self, args: &[Type<C>]) -> (usize, ImplStatus<C>) {
    if let Some((morph_index, fn_impl)) = self.find(args) {
      (morph_index, fn_impl.status().clone())
    } else {
      (0, ImplStatus::NotStarted)
    }
  }

  pub fn completed_at(&self, ind: usize) -> ImplRef<T, C> { self.completed.get(ind).unwrap().1.inner() }
  pub fn status_at(&self, ind: usize) -> ImplStatus<C> { self.completed.get(ind).unwrap().1.status().clone() }

  pub fn pending_for(&mut self, args: &[Type<C>], pass: &mut Pass<C>) -> (usize, ImplRef<T, C>) {
    self.set_pending(true);

    assert!(self.find_completed(args).is_none());

    // Prematurely put pass_2 into Status::Completed, so that it and its morph_index exists and can be retrieved
    // and used recursively.
    if pass.is_two() {
      let (args, fn_impl) =
        self.building.remove_entry(args).unwrap_or_else(|| panic!("Can't pending on missing impl for {:?}", args));
      self.completed.push((args, fn_impl));
      (self.completed.len() - 1, self.completed.last_mut().unwrap().1.pending(pass))
    } else if let Some(fn_impl) = self.find_building_mut(args) {
      (0, fn_impl.pending(pass))
    } else {
      self.building.insert(args.to_vec(), CallableImpl::new());
      (0, self.building.get_mut(args).unwrap().pending(pass))
    }
  }

  pub fn mark_done(&mut self, args: &[Type<C>], status: ImplStatus<C>, data: T::Impl) {
    // A simple boolean pending check will work here (instead of a stack or count), since we never enter a
    // recursive call.
    self.set_pending(false);

    // Sanity check to make sure it's properly moved over.
    assert!(status.is_completed() || self.find_completed(args).is_none());

    if !status.is_completed() {
      let building = self.find_building_mut(args).unwrap();
      building.set_status(status);
      building.set_data(data);
    } else {
      // We don't have change status here, since that was flipped in `pending_for`.
      self.find_completed_mut(args).unwrap().set_data(data);
    }
  }
}

/// A monomorphized function for a particular set of type arguments.
#[derive(Debug)]
pub struct CallableImpl<T: Callable<C>, C: Custom> {
  status: ImplStatus<C>,
  inner: Arc<Mutex<CallableImplInner<T, C>>>
}

impl<T: Callable<C>, C: Custom> CallableImpl<T, C> {
  pub fn new() -> CallableImpl<T, C> {
    CallableImpl { status: ImplStatus::NotStarted, inner: Arc::new(Mutex::new(CallableImplInner::new())) }
  }

  pub fn from(status: ImplStatus<C>, inner: Arc<Mutex<CallableImplInner<T, C>>>) -> CallableImpl<T, C> {
    CallableImpl { status, inner }
  }

  pub fn status(&self) -> &ImplStatus<C> { &self.status }
  pub fn status_mut(&mut self) -> &mut ImplStatus<C> { &mut self.status }
  pub fn set_status(&mut self, status: ImplStatus<C>) { self.status = status; }
  pub fn status_type(&self) -> Option<Type<C>> { self.status.status_type() }
  pub fn inner(&self) -> ImplRef<T, C> { ImplRef::new(self.inner.clone()) }
  pub fn set_data(&self, data: T::Impl) { self.inner.try_lock().unwrap().set_data(data); }

  pub fn pending(&mut self, pass: &mut Pass<C>) -> ImplRef<T, C> {
    if self.status.is_unstarted() {
      assert!(pass.is_one());
      self.set_status(ImplStatus::Pending);
    } else if pass.is_two() {
      let status = self.status_mut();
      if let ImplStatus::Discovered(t) = status {
        assert!(!t.is_unknown());
        *status = ImplStatus::Completed(std::mem::take(t));
      } else {
        panic!("status into pass 2 is not discovered");
      }
    }

    self.inner()
  }
}

pub struct ImplRef<T: Callable<C>, C: Custom> {
  inner: Arc<Mutex<CallableImplInner<T, C>>>
}

impl<T: Callable<C>, C: Custom> ImplRef<T, C> {
  pub fn new(inner: Arc<Mutex<CallableImplInner<T, C>>>) -> ImplRef<T, C> { ImplRef { inner } }
  pub fn borrow(&self) -> BorrowedImpl<T, C> { BorrowedImpl::Borrowed(self.inner.try_lock().unwrap()) }
}

pub enum BorrowedImpl<'i, T: Callable<C>, C: Custom> {
  Owned(CallableImplInner<T, C>),
  Borrowed(MutexGuard<'i, CallableImplInner<T, C>>)
}

impl<'i, T: Callable<C>, C: Custom> BorrowedImpl<'i, T, C> {
  pub fn borrowed(guard: MutexGuard<'i, CallableImplInner<T, C>>) -> BorrowedImpl<'i, T, C> {
    BorrowedImpl::Borrowed(guard)
  }

  pub fn item(&self) -> &CallableImplInner<T, C> {
    match self {
      Self::Owned(i) => i,
      Self::Borrowed(i) => i
    }
  }

  pub fn item_mut(&mut self) -> &mut CallableImplInner<T, C> {
    match self {
      Self::Owned(i) => i,
      Self::Borrowed(i) => i
    }
  }

  pub fn set_data(&mut self, data: T::Impl) { self.item_mut().set_data(data) }
  pub fn data(&self) -> &T::Impl { self.item().data() }
}

#[derive(Debug, Clone)]
pub enum ImplStatus<C: Custom> {
  /// The function has started into pass 2.
  Completed(Type<C>),

  /// The function type has been discovered on the first pass, but not written on the second pass.
  Discovered(Type<C>),

  /// The function type is in the process of being discovered.
  Pending,

  /// Function discovery has not yet been attempted.
  NotStarted
}

impl<C: Custom> ImplStatus<C> {
  pub fn status_type(&self) -> Option<Type<C>> {
    match self {
      Self::Completed(t) | Self::Discovered(t) => Some(t.clone()),
      _ => None
    }
  }

  pub fn is_completed(&self) -> bool { matches!(self, Self::Completed(_)) }
  pub fn is_discovered(&self) -> bool { matches!(self, Self::Discovered(_)) }
  pub fn is_unstarted(&self) -> bool { matches!(self, Self::NotStarted) }
}

#[derive(Debug)]
pub struct CallableImplInner<T: Callable<C>, C: Custom> {
  data: T::Impl
}

impl<T: Callable<C>, C: Custom> CallableImplInner<T, C> {
  pub fn new() -> CallableImplInner<T, C> { CallableImplInner { data: Default::default() } }
  pub fn set_data(&mut self, data: T::Impl) { self.data = data; }
  pub fn data(&self) -> &T::Impl { &self.data }
}

////////////////////////////////////////////////////////////////////////
// Code-defined functions

/// The specific callable that is built from Glider source code.
#[derive(Debug, Clone)]
pub struct CodeFn<C: Custom> {
  params: Vec<Destructure>,
  body: Block<Captured>,
  captures: Vec<(String, Type<C>)>
}

impl<C: Custom> CodeFn<C> {
  pub fn new(params: Vec<Destructure>, body: Block<Captured>, captures: Vec<(String, Type<C>)>) -> CodeFn<C> {
    CodeFn { params, body, captures }
  }

  pub fn params(&self) -> &[Destructure] { &self.params }
  pub fn body(&self) -> &Block<Captured> { &self.body }
  pub fn captures(&self) -> &[(String, Type<C>)] { &self.captures }
  pub fn set_captures(&mut self, captures: Vec<(String, Type<C>)>) { self.captures = captures; }
}

#[async_trait]
impl<C: Custom> Callable<C> for CodeFn<C> {
  type Inner = CodeRunData;
  type Impl = Chunk<C>;

  fn extracts_and_args(def: &FnDef<C>, args: &[Type<C>]) -> (Vec<Extract>, Vec<Type<C>>) {
    let extracts = find_extracts(def, args);
    let args: Vec<_> = extracts.iter().zip(args.iter()).flat_map(|(e, a)| e.extracted_refs(a)).cloned().collect();
    (extracts, args)
  }

  async fn unpack_args<'p>(
    fn_def: &FnDef<C>, compiler: &mut Compiler<'p, C>, args: &[ExpressionMeta<Captured>]
  ) -> Vec<Type<C>> {
    // We put the destructured arguments on the stack just before making the call in order to make sure they get
    // aligned correctly, but we won't mark them as initialized variables until we're inside the call.
    compiler.assignments_unmarked(fn_def.params(), args).await
  }

  async fn compile(
    fn_def: &FnDef<C>, _pending: ImplRef<Self, C>, args: &[Type<C>], pass: &mut Pass<C>
  ) -> (Type<C>, Chunk<C>) {
    let compiler = Compiler::new(pass);
    compiler.compile_fn(fn_def, args).await
  }
}

pub struct CodeRunData {
  pos: Position
}

impl CodeRunData {
  pub fn new(pos: Position) -> CodeRunData { CodeRunData { pos } }
  pub fn pos(&self) -> &Position { &self.pos }
}

/// The source-code-based callable definition for Glider source code.
pub type FnDef<C> = CallableDef<CodeFn<C>, C>;

impl<C: Custom> fmt::Debug for FnDef<C> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.inner.try_lock().unwrap().fmt(f) }
}

impl<C: Custom> FnDef<C> {
  pub fn new(
    params: Vec<Destructure>, body: Block<Captured>, captures: Vec<(String, Type<C>)>, pos: Position
  ) -> FnDef<C> {
    FnDef {
      def: CodeFn::new(params, body, captures),
      inner: Arc::new(Mutex::new(FnDefInner::new(CodeRunData::new(pos))))
    }
  }

  pub fn is_unknown(&self) -> bool { self.captures().iter().any(|(_, t)| t.is_unknown()) }
  pub fn pos(&self) -> Position { self.inner.try_lock().unwrap().pos().clone() }
  pub fn params(&self) -> &[Destructure] { self.def.params() }
  pub fn body(&self) -> &Block<Captured> { self.def.body() }
  pub fn captures(&self) -> &[(String, Type<C>)] { self.def.captures() }
  pub fn set_captures(&mut self, captures: Vec<(String, Type<C>)>) { self.def.set_captures(captures); }

  pub fn similar(&self, other: &FnDef<C>) -> bool {
    // self == other || self.inner().try_lock().unwrap().pos() == other.inner().try_lock().unwrap().pos()
    self == other || Arc::ptr_eq(&self.inner(), &other.inner())
  }
}

pub type FnDefInner<C> = CallableDefInner<CodeFn<C>, C>;

impl<C: Custom> fmt::Debug for FnDefInner<C> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "fn() at {}", self.pos()) }
}

impl<C: Custom> FnDefInner<C> {
  pub fn pos(&self) -> &Position { self.def.pos() }
}

pub type FnImpl<C> = CallableImpl<CodeFn<C>, C>;

impl<C: Custom> Default for FnImpl<C> {
  fn default() -> FnImpl<C> { FnImpl::new() }
}

pub type FnBorrowedImpl<'i, C> = BorrowedImpl<'i, CodeFn<C>, C>;

impl<'i, C: Custom> FnBorrowedImpl<'i, C> {
  pub fn chunk(&self) -> &Chunk<C> { self.item().chunk() }
  pub fn chunk_mut(&mut self) -> &mut Chunk<C> { self.item_mut().chunk_mut() }
  pub fn emit(&mut self, op: Opcode, pos: Position) { self.item_mut().emit(op, pos); }
  pub fn add_constant(&mut self, c: Constant<C>) -> usize { self.chunk_mut().add_constant(c) }
  pub fn add_opcode(&mut self, op: Opcode, pos: Position) { self.chunk_mut().add_opcode(op, pos); }
  pub fn patch_jump(&mut self, ind: usize, val: usize) { self.chunk_mut().patch_jump(ind, val); }

  pub fn constants(&self) -> &Constants<C> { self.chunk().constants() }
  pub fn code(&self) -> &[Opcode] { self.chunk().code() }
  pub fn positions(&self) -> &[Position] { self.chunk().positions() }
  pub fn code_len(&self) -> usize { self.chunk().code_len() }
  pub fn at(&self, ip: usize) -> Option<&Opcode> { self.chunk().at(ip) }
  pub fn constant_at(&self, ind: usize) -> Option<&Constant<C>> { self.chunk().constant_at(ind) }
}

pub type FnImplInner<C> = CallableImplInner<CodeFn<C>, C>;

impl<C: Custom> Default for FnImplInner<C> {
  fn default() -> FnImplInner<C> { FnImplInner::new() }
}

impl<C: Custom> FnImplInner<C> {
  pub fn chunk(&self) -> &Chunk<C> { &self.data }
  pub fn chunk_mut(&mut self) -> &mut Chunk<C> { &mut self.data }
  pub fn emit(&mut self, op: Opcode, pos: Position) { self.data.add_opcode(op, pos); }
}

////////////////////////////////////////////////////////////////////////
// Native-defined functions

pub type NativeBuildFn<C> = for<'p> fn(
  Vec<Type<C>>,
  Vec<Type<C>>,
  &'p mut Pass<C>
) -> Pin<Box<dyn Future<Output = (Type<C>, BuildData<C>)> + Send + 'p>>;

pub type NativeRunFn<C> =
  fn(Vec<Value<C>>, Vec<Value<C>>, BuildData<C>, RunData<C>) -> Pin<Box<dyn Future<Output = Value<C>> + Send>>;

#[derive(Clone)]
pub struct NativeFn<C: Custom> {
  build_fn: NativeBuildFn<C>,
  run_fn: NativeRunFn<C>,
  captures: Vec<(String, Type<C>)>
}

impl<C: Custom> fmt::Debug for NativeFn<C> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "<native>") }
}

#[async_trait]
impl<C: Custom> Callable<C> for NativeFn<C> {
  type Inner = NativeRunData<C>;
  type Impl = BuildData<C>;

  fn extracts_and_args(_def: &NativeDef<C>, args: &[Type<C>]) -> (Vec<Extract>, Vec<Type<C>>) {
    (vec![Extract::solo(); args.len()], args.to_vec())
  }

  async fn unpack_args<'p>(
    _this: &CallableDef<Self, C>, compiler: &mut Compiler<'p, C>, args: &[ExpressionMeta<Captured>]
  ) -> Vec<Type<C>> {
    let mut types = Vec::new();
    for a in args {
      types.push(compiler.expression(a).await);
    }
    types
  }

  async fn compile(
    this: &NativeDef<C>, _pending: ImplRef<Self, C>, args: &[Type<C>], pass: &mut Pass<C>
  ) -> (Type<C>, BuildData<C>) {
    // TODO(performance): don't `args.to_vec()`: caller might be able to move.
    let capts = this.def.captures.iter().map(|c| c.1.clone()).collect();

    (this.def.build_fn)(args.to_vec(), capts, pass).await
  }
}

impl<C: Custom> NativeFn<C> {
  pub fn new(build_fn: NativeBuildFn<C>, run_fn: NativeRunFn<C>) -> NativeFn<C> {
    NativeFn { build_fn, run_fn, captures: Vec::new() }
  }

  pub fn build_fn(&self) -> &NativeBuildFn<C> { &self.build_fn }
  pub fn run_fn(&self) -> &NativeRunFn<C> { &self.run_fn }
  pub fn captures(&self) -> &[(String, Type<C>)] { &self.captures }
  pub fn set_captures(&mut self, captures: Vec<(String, Type<C>)>) { self.captures = captures; }
}

pub struct NativeRunData<C: Custom> {
  run_fn: NativeRunFn<C>
}

impl<C: Custom> PartialEq for NativeRunData<C> {
  fn eq(&self, other: &NativeRunData<C>) -> bool {
    // TODO: this function should actually be named `similar`, since it doesn't match capture vals.
    self.run_fn as usize == other.run_fn as usize
  }
}

impl<C: Custom> Eq for NativeRunData<C> {}

impl<C: Custom> NativeRunData<C> {
  pub fn new(run_fn: NativeRunFn<C>) -> NativeRunData<C> { NativeRunData { run_fn } }

  pub fn run_fn(&self) -> &NativeRunFn<C> { &self.run_fn }
}

pub type NativeDef<C> = CallableDef<NativeFn<C>, C>;

impl<C: Custom> fmt::Debug for NativeDef<C> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { self.inner.try_lock().unwrap().fmt(f) }
}

impl<C: Custom> NativeDef<C> {
  pub fn new(build_fn: NativeBuildFn<C>, run_fn: NativeRunFn<C>) -> NativeDef<C> {
    NativeDef {
      def: NativeFn::new(build_fn, run_fn),
      inner: Arc::new(Mutex::new(NativeDefInner::new(NativeRunData::new(run_fn))))
    }
  }

  pub fn native_fn(&self) -> &NativeFn<C> { &self.def }
  pub fn build_fn(&self) -> &NativeBuildFn<C> { self.native_fn().build_fn() }
  pub fn run_fn(&self) -> &NativeRunFn<C> { self.native_fn().run_fn() }
  pub fn is_unknown(&self) -> bool { self.captures().iter().any(|(_, t)| t.is_unknown()) }
  pub fn captures(&self) -> &[(String, Type<C>)] { &self.def.captures }
  pub fn set_captures(&mut self, capts: Vec<(String, Type<C>)>) { self.def.captures = capts; }

  pub fn similar(&self, other: &NativeDef<C>) -> bool {
    self == other || self.inner().try_lock().unwrap().similar(&other.inner().try_lock().unwrap())
  }
}

pub type NativeDefInner<C> = CallableDefInner<NativeFn<C>, C>;

impl<C: Custom> fmt::Debug for NativeDefInner<C> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "<native>") }
}

impl<C: Custom> NativeDefInner<C> {
  pub fn run_fn(&self) -> &NativeRunFn<C> { self.def.run_fn() }
  pub fn similar(&self, other: &NativeDefInner<C>) -> bool { self.def == other.def }
}

pub type NativeBorrowedImpl<'i, C> = BorrowedImpl<'i, NativeFn<C>, C>;

pub type NativeImplInner<C> = CallableImplInner<NativeFn<C>, C>;

impl<C: Custom> Default for NativeImplInner<C> {
  fn default() -> NativeImplInner<C> { NativeImplInner::new() }
}

/// The native analog of a chunk, which is implementation-specific execution details.
///
/// Build data include: the indices of functions that it needs to call; generated functions that it wants to
/// return; and object maps that are used by the native function.
#[derive(Clone, Debug, Default)]
pub struct BuildData<C: Custom> {
  obj_maps: Vec<ObjMap>,
  fn_inds: Vec<FnInd>,
  native_gens: Vec<Arc<Mutex<NativeDefInner<C>>>>
}

impl<C: Custom> BuildData<C> {
  pub fn new() -> BuildData<C> { BuildData { obj_maps: Vec::new(), fn_inds: Vec::new(), native_gens: Vec::new() } }
  pub fn push_obj_map(&mut self, map: ObjMap) { self.obj_maps.push(map); }
  pub fn push_fn_ind(&mut self, ind: FnInd) { self.fn_inds.push(ind); }
  pub fn push_native_gen(&mut self, gen: Arc<Mutex<NativeDefInner<C>>>) { self.native_gens.push(gen); }

  pub fn obj_maps(&self) -> &[ObjMap] { &self.obj_maps }
  pub fn fn_inds(&self) -> &[FnInd] { &self.fn_inds }
  pub fn native_gens(&self) -> &[Arc<Mutex<NativeDefInner<C>>>] { &self.native_gens }
  pub fn into_fn_inds(self) -> Vec<FnInd> { self.fn_inds }

  pub fn with_fn_ind(mut self, ind: FnInd) -> Self {
    self.push_fn_ind(ind);
    self
  }

  pub fn with_obj_map(mut self, om: ObjMap) -> Self {
    self.push_obj_map(om);
    self
  }

  pub fn with_native_gen(mut self, gen: Arc<Mutex<NativeDefInner<C>>>) -> Self {
    self.push_native_gen(gen);
    self
  }
}

/// Runtime information that is additionally passed to native functions.
///
/// Run data include: the call-site code position; the status of the current native call stack; and any
/// external capture object for this implementation.
pub struct RunData<C: Custom> {
  pos: Position,
  status: C::Status,
  capture: C::Capture
}

impl<C: Custom> RunData<C> {
  pub fn new(pos: Position, status: C::Status, capture: C::Capture) -> RunData<C> { RunData { pos, status, capture } }
  pub fn pos(&self) -> &Position { &self.pos }
  pub fn status(&self) -> &C::Status { &self.status }

  /// External captures work very similar to normal captures, except that they aren't necessarily runtime
  /// values, and therefore have no `Type` analog in the build function. These are used primarily to allow
  /// native functions to be injected with arbitrary captured values.
  pub fn capture(&self) -> &C::Capture { &self.capture }
}

// TODO(later): Nest object maps for nested objects/arrays.
#[derive(Clone, Debug)]
pub enum ObjMap {
  Unmapped,
  Spec(Vec<usize>),
  Obj(Vec<(String, ObjMap)>),
  Arr(Vec<ObjMap>)
}

impl ObjMap {
  pub fn for_object<C: Custom>(obj: &Object<C>, keys: &[&str]) -> ObjMap {
    ObjMap::Spec(keys.iter().map(|k| obj.index_of(k).unwrap()).collect())
  }

  pub fn full<C: Custom>(obj: &Type<C>) -> ObjMap {
    if let Type::Object(o) = obj {
      ObjMap::Obj(o.array().iter().map(|(s, t)| (s.clone(), ObjMap::full(t))).collect())
    } else if let Type::Array(a) = obj {
      ObjMap::Arr(a.types().iter().map(|t| ObjMap::full(t)).collect())
    } else {
      ObjMap::Unmapped
    }
  }

  pub fn for_unit() -> ObjMap { ObjMap::Unmapped }

  pub fn read<'v, C: Custom>(&self, array: &'v mut [Value<C>], ind: usize) -> &'v mut Value<C> {
    // You can only do a _read_ when you have a specified object map.
    match self {
      Self::Unmapped => &mut array[ind],
      Self::Spec(refs) => &mut array[refs[ind]],
      Self::Obj(_) => panic!("Unspecified obj map"),
      Self::Arr(_) => panic!("Unspecified arr map")
    }
  }
}
