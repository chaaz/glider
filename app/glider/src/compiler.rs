//! The compiler (3rd pass) for Glider, which converts an enhanced AST into some bytecode.
//!
//! This compiler is itself a localized multi-pass mono-morphizing compiler in order to ensure type correctness
//! via monophorphization for each call site. It first compiles the function to determine its type, and then
//! recompiles it to generate its bytecode. Consider function like this (direct recursive functions aren't yet
//! allowed by the language, but let's pretend):
//!
//! ```glider
//! recurse = fn(v) {
//!   = if v < 5 {
//!     recurse(v + 1) + 2
//!   } else {
//!     1
//!   }
//! }
//! ```
//!
//! We must first compile this to determine that the return is an numeric type (by skipping the recursive case
//! and analyzing the base case). Once we're sure that the type `recurse(v + 1)` is numeric,
//! the second pass can generate the bytecode for numeric addition for the `+ 2` expression.
//!
//! Footnotes:
//!
//! (1) TODO(performance, maybe): by uncritically pass 1 compiling a function based only on if it has a known
//! type, we risk multiple unnecessary compilations: in the worst case, `d^2` (where `d` is its call depth). To
//! alleviate this, we could also keep a dependency tree in the fn_impl which describes which direct fn_impls
//! need to be discovered first; and then only recompile when that tree is satisfied.
//!
//! The case might be ignorable, however: it's possible that the worst case happens when there is a
//! corresponding worst case runtime, but I haven't found such an example. The worst case compile time **does**
//! happens if there's an infinite recursion: below, we have to pass 1 `c` multiple times from pass 1 `b`. But
//! poor performance detecting non-termination might be ok.
//!
//! ```glider
//! a = fn(n) { if n > 1 { b(n) } else { 1 } };
//! b = fn(n) { c(n) };
//! c = fn(n) { a(n - 1) + b(n - 1) };  // infinite recursion!
//! a(3)
//! ```
//!
//! (2) By using a `pending` check at the fn_def level in pass 1, we ensure that no recursion is breached, even
//! recursion to other impls of the same fn_def (but with different type args).
//!
//! ```glider
//! f = fn(a) { if very_strange(a) { f(push(a, 1)) } else { len(a) } };
//! f([])
//! ```
//!
//! In theory, we **should** compile f([]), f([int]), f([int, int]), etc, and figure out a way to stop
//! termination. Maybe. Somehow. But instead we refuse to enter `f([int])`, since `f` is pending from `f([])`.
//! This means that the compiler can't recurse into different implementations of the same function, which is a
//! limitation that we accept for now.
//!
//! (3) Assume at the start of a compile pass:
//!
//! - fn_def is not borrowed, but it may be pending. We may need to briefly borrow to get or set its pending
//!   status.
//! - fn_impl might be borrowed if fn_def is pending; this happens if we're currently recursively calling the
//!   function.
//! - if fn_impl is borrowed, because the fn_impl is in pass_2, we'll already have a known type, which can be
//!   obtained without borrowing.

use crate::callable::{Callable, CallableDef, FnDef, ImplStatus, NativeBuildFn, NativeDef, NativeRunFn};
use crate::custom::Custom;
use crate::enhancer::Capture;
use crate::enhancer::Captured;
use crate::errors::Context;
use crate::parser::{Assignment, BinaryOp, Block, Destructure, DotIndex, Expression, ExpressionMeta, Scope, Script,
                    UnaryOp};
use crate::scanner::Position;
use crate::types::{Array, Object, Type};
use crate::value::{Constant, Literal, Value};
use async_recursion::async_recursion;
use std::fmt;

pub async fn compile<C: Custom>(code: Script<Captured>) -> FnDef<C> { compile_with_natives(code, &[]).await.0 }

pub async fn compile_with_natives<C: Custom>(
  code: Script<Captured>, natives: &[(NativeBuildFn<C>, NativeRunFn<C>, &str)]
) -> (FnDef<C>, Vec<Value<C>>) {
  let mut script = FnDef::new(Vec::new(), code.into_block(), Vec::new(), Position::zero());
  let native_vals = add_natives(&mut script, natives);

  // Scripts could go directly to pass 2 since they're implicitly safe from recursion. However, we don't have
  // the ability to prime them (we can't create a pending impl with status `Discovered(Type)`, because we don't
  // yet know what the type is). So, we'll just double-pass, just like every other function.

  let mut one = Pass::one();
  let (_zero, _script_type) = script.compile(&[], &mut one).await;
  let (_zero, _script_type) = script.compile(&[], one.two()).await;
  (script, native_vals)
}

pub struct Compiler<'p, C: Custom> {
  pass: &'p mut Pass<C>,
  chunk: Chunk<C>,
  scope: Scope<Type<C>>
}

impl<'p, C: Custom> Compiler<'p, C> {
  pub fn new(pass: &'p mut Pass<C>) -> Compiler<'p, C> { Compiler { pass, chunk: Chunk::new(), scope: Scope::new() } }

  pub fn chunk(&self) -> &Chunk<C> { &self.chunk }
  pub fn chunk_mut(&mut self) -> &mut Chunk<C> { &mut self.chunk }
  pub fn pass(&self) -> &Pass<C> { self.pass }
  pub fn scope_mut(&mut self) -> &mut Scope<Type<C>> { &mut self.scope }

  pub async fn compile_fn(mut self, fn_def: &FnDef<C>, dstr_args: &[Type<C>]) -> (Type<C>, Chunk<C>) {
    assert_eq!(fn_def.params().iter().map(|d| d.identifier_count()).sum::<usize>(), dstr_args.len());
    self.begin_scope();

    for (id, tp) in fn_def.params().iter().flat_map(|p| p.identifiers()).zip(dstr_args) {
      self.declare_variable(id.to_string());
      self.mark_initialized(1, tp.clone());
    }

    for (id, tp) in fn_def.captures() {
      self.declare_variable(id.to_string());
      self.mark_initialized(1, tp.clone());
    }

    let block_type = self.block(fn_def.body(), false, false).await;
    self.emit(Opcode::Return, Position::zero());

    self.end_scope();

    (block_type, self.chunk)
  }

  pub async fn block(&mut self, block: &Block<Captured>, conditional: bool, last_conditional: bool) -> Type<C> {
    self.begin_scope();

    for asgn in block.assignments() {
      self.assignment(asgn).await;
    }
    let etype = self.expression(block.value()).await;

    if conditional {
      self.reserve_used();
      if last_conditional {
        self.restore_used();
      }
    }

    self.end_scope();
    etype
  }

  pub async fn assignment(&mut self, asgn: &Assignment<Captured>) {
    self.declare_all_variables(asgn.lhs());
    let rtype = self.expression(asgn.rhs()).await;
    let extract = self.find_extract(asgn.lhs(), &rtype, true);

    if extract.is_empty() {
      self.emit(Opcode::Pop, asgn.pos().clone());
    } else if !extract.is_solo() {
      self.emit(Opcode::Extract(extract), asgn.pos().clone());
    }
  }

  /// Traverse each destructure:argument pair as if it was an assignment, except don't record or mark the
  /// variable in the current scope.
  pub async fn assignments_unmarked(&mut self, lhs: &[Destructure], args: &[ExpressionMeta<Captured>]) -> Vec<Type<C>> {
    assert_eq!(lhs.len(), args.len());

    let mut collector = Vec::new();
    for (d, a) in lhs.iter().zip(args.iter()) {
      let atype = self.expression(a).await;
      let extract = self.find_extract(d, &atype, false);

      if extract.is_empty() {
        self.emit(Opcode::Pop, a.pos().clone());
      } else if !extract.is_solo() {
        #[allow(clippy::needless_collect)] // To unify if/else
        let mut dstr_types: Vec<_> = extract.extracted_types(atype).collect();
        self.emit(Opcode::Extract(extract), a.pos().clone());
        collector.append(&mut dstr_types);
      } else {
        collector.push(atype);
      }
    }

    collector
  }

  pub fn declare_all_variables(&mut self, dstr: &Destructure) {
    struct Declare;
    impl<C: Custom> DestructVisitor<C> for Declare {
      fn visit_identifier(&mut self, compiler: &mut Compiler<C>, id: &str) {
        compiler.declare_variable(id.to_string());
      }
    }
    Declare.visit_destructure(self, dstr);
  }

  pub fn find_extract(&mut self, lhs: &Destructure, rtype: &Type<C>, mark: bool) -> Extract {
    struct Extractor<'t, C: Custom> {
      mark: bool,
      type_stack: Vec<&'t Type<C>>,
      indexes: Vec<Vec<usize>>,
      active_index: Vec<usize>,
      init_mark: usize
    }
    impl<'t, C: Custom> DestructVisitor<C> for Extractor<'t, C> {
      fn visit_identifier(&mut self, compiler: &mut Compiler<C>, _id: &str) {
        self.indexes.push(self.active_index.clone());
        if self.mark {
          compiler.mark_initialized(self.init_mark, (*self.type_stack.last().unwrap()).clone());
        }
        self.init_mark -= 1;
      }

      fn visit_array(&mut self, compiler: &mut Compiler<C>, array: &[Destructure]) {
        for (i, d) in array.iter().enumerate() {
          self.active_index.push(i);
          self.type_stack.push(self.type_stack.last().unwrap().as_array().get(i));
          self.visit_destructure(compiler, d);
          self.type_stack.pop();
          self.active_index.pop();
        }
      }

      fn visit_object(&mut self, compiler: &mut Compiler<C>, object: &[(String, Destructure)]) {
        for (s, d) in object {
          let obj_type = self.type_stack.last().unwrap().as_object();
          let i = obj_type.index_of(s).unwrap();
          self.active_index.push(i);
          self.type_stack.push(obj_type.get(s.as_str()));
          self.visit_destructure(compiler, d);
          self.type_stack.pop();
          self.active_index.pop();
        }
      }
    }

    let mut extractor = Extractor {
      mark,
      type_stack: vec![rtype],
      indexes: Vec::new(),
      active_index: Vec::new(),
      init_mark: lhs.identifier_count()
    };
    extractor.visit_destructure(self, lhs);

    Extract { indexes: extractor.indexes }
  }

  #[async_recursion]
  pub async fn expression(&mut self, expr: &ExpressionMeta<Captured>) -> Type<C> {
    match expr.expr() {
      Expression::Variable(name) => self.variable(name, expr),
      Expression::Literal(v) => self.literal(v, expr),
      Expression::Unary(op, expr) => self.unary(op, expr).await,
      Expression::Binary(op, lexpr, rexpr) => self.binary(*op, lexpr, rexpr).await,
      Expression::Array(exprs) => self.array(exprs, expr).await,
      Expression::Object(pairs) => self.object(pairs, expr).await,
      Expression::Dot(expr, dot_ind) => self.dot(expr, dot_ind).await,
      Expression::Index(expr, ind) => self.index(expr, *ind).await,
      Expression::IfBlock(cond, block, elseifs, last) => self.if_block(cond, block, elseifs, last).await,
      Expression::And(expr1, expr2) => self.and(expr1, expr2).await,
      Expression::Or(expr1, expr2) => self.or(expr1, expr2).await,
      Expression::Function(params, block, capture) => self.fn_def(params, block, capture, expr.pos()),
      Expression::Call(expr, args) => self.call(expr, args).await
    }
  }

  fn variable(&mut self, name: &str, expr: &ExpressionMeta<Captured>) -> Type<C> {
    if let Some((i, vtype)) = self.resolve_local(name) {
      self.emit(Opcode::GetLocal(i), expr.pos().clone());
      vtype
    } else {
      panic!("Couldn't find variable with name \"{}\".", name);
    }
  }

  fn literal(&mut self, lit: &Literal, expr: &ExpressionMeta<Captured>) -> Type<C> {
    let (vtype, cnst) = match lit {
      Literal::Unit => (Type::Unit, Constant::Unit),
      Literal::String(s) => (Type::String(Some(s.to_string())), Constant::String(s.clone())),
      Literal::Int(v) => (Type::Number((*v).into()), Constant::Int(*v)),
      Literal::Float(v) => (Type::Number((*v).into()), Constant::Float(*v)),
      Literal::Bool(v) => (Type::Bool(Some(*v)), Constant::Bool(*v))
    };
    let const_ind = self.add_constant(cnst);
    self.emit(Opcode::Constant(const_ind), expr.pos().clone());
    vtype
  }

  async fn unary(&mut self, op: &UnaryOp, expr: &ExpressionMeta<Captured>) -> Type<C> {
    let etype = self.expression(expr).await;
    match op {
      UnaryOp::Negate => {
        assert!(etype.is_number() || etype.is_unknown());
        self.emit(Opcode::Negate, expr.pos().clone());
      }
      UnaryOp::Not => {
        assert!(etype.is_bool() || etype.is_unknown());
        self.emit(Opcode::Not, expr.pos().clone());
      }
    }
    etype
  }

  async fn binary(
    &mut self, op: BinaryOp, lexpr: &ExpressionMeta<Captured>, rexpr: &ExpressionMeta<Captured>
  ) -> Type<C> {
    // TODO(later, shortcut): shortcut w/ constexprs if possible. See `Pass::define()` if that gets implemented.

    let ltype = self.expression(lexpr).await;
    let rtype = self.expression(rexpr).await;

    // Allowed type combinations.
    assert!(
      (ltype.is_unknown() || rtype.is_unknown())
        || (ltype.is_json() || rtype.is_json())
        || (ltype.is_string() && rtype.is_string())
        || ltype == rtype
    );

    // Allowed operations.
    match op {
      BinaryOp::Subtract
      | BinaryOp::Multiply
      | BinaryOp::Divide
      | BinaryOp::Mod
      | BinaryOp::Gt
      | BinaryOp::Gte
      | BinaryOp::Lt
      | BinaryOp::Lte => {
        assert!(ltype.is_unknown() || ltype.is_json() || ltype.is_number())
      }
      BinaryOp::Add => {
        assert!(ltype.is_unknown() || ltype.is_json() || ltype.is_string() || ltype.is_number())
      }
      BinaryOp::Equals | BinaryOp::NotEquals => {
        assert!(ltype.is_unknown() || ltype.is_json() || ltype.is_string() || ltype.is_number() || ltype.is_bool())
      }
    }

    // Emit the code.
    let opcode = match op {
      BinaryOp::Add => Opcode::Add,
      BinaryOp::Subtract => Opcode::Subtract,
      BinaryOp::Multiply => Opcode::Multiply,
      BinaryOp::Divide => Opcode::Divide,
      BinaryOp::Mod => Opcode::Mod,
      BinaryOp::Gt => Opcode::Gt,
      BinaryOp::Gte => Opcode::Gte,
      BinaryOp::Lt => Opcode::Lt,
      BinaryOp::Lte => Opcode::Lte,
      BinaryOp::Equals => Opcode::Equals,
      BinaryOp::NotEquals => Opcode::NotEquals
    };
    self.emit(opcode, lexpr.pos().clone());

    // Determine the return type.
    if ltype.is_unknown() || rtype.is_unknown() {
      Type::Unknown
    } else if ltype.is_string() {
      // special string types
      if ltype.is_string_literal() && rtype.is_string_literal() && op == BinaryOp::Add {
        Type::String(Some(format!("{}{}", ltype.as_string_literal(), rtype.as_string_literal())))
      } else if rtype.is_string() {
        Type::String(None)
      } else {
        Type::Json
      }
    } else if !ltype.is_json() && !rtype.is_json() {
      // known type combinations
      match op {
        BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide | BinaryOp::Mod => ltype,
        _ => Type::Bool(None)
      }
    } else {
      // JSON combinations
      match op {
        BinaryOp::Add | BinaryOp::Subtract | BinaryOp::Multiply | BinaryOp::Divide | BinaryOp::Mod => Type::Json,
        _ => Type::Bool(None)
      }
    }
  }

  async fn array(&mut self, exprs: &[ExpressionMeta<Captured>], expr: &ExpressionMeta<Captured>) -> Type<C> {
    let mut types = Vec::new();
    for expr in exprs {
      types.push(self.expression(expr).await);
    }

    self.emit(Opcode::Array(types.len()), expr.pos().clone());
    Type::Array(Box::new(Array::new(types)))
  }

  async fn object(&mut self, pairs: &[(String, ExpressionMeta<Captured>)], expr: &ExpressionMeta<Captured>) -> Type<C> {
    let mut array = Vec::new();
    for (k, e) in pairs {
      array.push((k.clone(), self.expression(e).await));
    }
    array.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));
    let order: Vec<_> = array.iter().map(|(o, _)| pairs.iter().position(|(k, _)| k == o).unwrap()).collect();

    self.emit(Opcode::Object(order), expr.pos().clone());
    Type::Object(Box::new(Object::new(array)))
  }

  async fn dot(&mut self, expr: &ExpressionMeta<Captured>, dot_ind: &DotIndex) -> Type<C> {
    let ltype = self.expression(expr).await;
    match dot_ind {
      DotIndex::Identifier(s) => match ltype {
        Type::Unknown => Type::Unknown,
        Type::Object(o) => {
          let ind = o.index_of(s).unwrap_or_else(|| panic!("No index \"{}\" at {}", s, expr.pos()));
          self.emit(Opcode::GetIndex(ind), expr.pos().clone());
          o.get(s).clone()
        }
        Type::Json => {
          self.emit(Opcode::GetJsonKey(s.clone()), expr.pos().clone());
          Type::Json
        }
        other => panic!("Not indexable by identifier: {:?} at {}", other, expr.pos())
      },
      DotIndex::Integer(i) => {
        let i = *i as usize;
        match ltype {
          Type::Unknown => Type::Unknown,
          Type::Array(a) => {
            self.emit(Opcode::GetIndex(i), expr.pos().clone());
            a.get(i).clone()
          }
          Type::Json => {
            self.emit(Opcode::GetJsonIndex(i), expr.pos().clone());
            Type::Json
          }
          other => panic!("Not indexable by int: {:?} at {}", other, expr.pos())
        }
      }
    }
  }

  async fn index(&mut self, expr: &ExpressionMeta<Captured>, i: u64) -> Type<C> {
    let ltype = self.expression(expr).await;
    let i = i as usize;
    match ltype {
      Type::Unknown => Type::Unknown,
      Type::Array(a) => {
        self.emit(Opcode::GetIndex(i), expr.pos().clone());
        a.get(i).clone()
      }
      Type::Json => {
        self.emit(Opcode::GetJsonIndex(i), expr.pos().clone());
        Type::Json
      }
      other => panic!("Not indexable by int: {:?} at {}", other, expr.pos())
    }
  }

  async fn if_block(
    &mut self, cond: &ExpressionMeta<Captured>, block: &Block<Captured>,
    elseifs: &[(ExpressionMeta<Captured>, Block<Captured>)], last: &Block<Captured>
  ) -> Type<C> {
    // TODO(later, shortcut): shortcut w/ constexprs if possible. See `Pass::define()` if that gets implemented.

    let if_type = self.expression(cond).await;
    assert!(if_type.is_bool() || if_type.is_unknown());
    let false_jump = self.emit_jump(Opcode::initial_jump_if_false(), cond.pos().clone());
    self.emit(Opcode::Pop, cond.pos().clone());

    let mut b_types = vec![self.block(block, true, false).await];
    let done_jump = self.emit_jump(Opcode::initial_jump(), cond.pos().clone());
    self.patch_jump(false_jump);
    self.emit(Opcode::Pop, cond.pos().clone());

    let mut dx_jumps = Vec::new();
    for (cond, block) in elseifs {
      let if_type = self.expression(cond).await;
      assert!(if_type.is_bool() || if_type.is_unknown());
      let fx_jump = self.emit_jump(Opcode::initial_jump_if_false(), cond.pos().clone());
      self.emit(Opcode::Pop, cond.pos().clone());

      b_types.push(self.block(block, true, false).await);
      dx_jumps.push(self.emit_jump(Opcode::initial_jump(), cond.pos().clone()));
      self.patch_jump(fx_jump);
      self.emit(Opcode::Pop, cond.pos().clone());
    }

    b_types.push(self.block(last, true, true).await);

    self.patch_jump(done_jump);
    for jump in dx_jumps {
      self.patch_jump(jump);
    }

    Type::unify(&b_types)
  }

  async fn and(&mut self, left: &ExpressionMeta<Captured>, right: &ExpressionMeta<Captured>) -> Type<C> {
    // TODO(later, shortcut): shortcut w/ constexprs if possible. See `Pass::define()` if that gets implemented.

    let ltype = self.expression(left).await;
    assert!(ltype.is_bool() || ltype.is_unknown());

    let end_jump = self.emit_jump(Opcode::initial_jump_if_false(), left.pos().clone());
    self.emit(Opcode::Pop, left.pos().clone());

    let rtype = self.expression(right).await;
    assert!(rtype.is_bool() || rtype.is_unknown());

    self.patch_jump(end_jump);

    if ltype.is_unknown() || rtype.is_unknown() {
      Type::Unknown
    } else {
      Type::Bool(None)
    }
  }

  async fn or(&mut self, left: &ExpressionMeta<Captured>, right: &ExpressionMeta<Captured>) -> Type<C> {
    // TODO(later, shortcut): shortcut w/ constexprs if possible. See `Pass::define()` if that gets implemented.

    let ltype = self.expression(left).await;
    assert!(ltype.is_bool() || ltype.is_unknown());

    let else_jump = self.emit_jump(Opcode::initial_jump_if_false(), left.pos().clone());
    let end_jump = self.emit_jump(Opcode::initial_jump(), left.pos().clone());

    self.patch_jump(else_jump);
    self.emit(Opcode::Pop, left.pos().clone());

    let rtype = self.expression(right).await;
    assert!(rtype.is_bool() || rtype.is_unknown());

    self.patch_jump(end_jump);

    if ltype.is_unknown() || rtype.is_unknown() {
      Type::Unknown
    } else {
      Type::Bool(None)
    }
  }

  fn fn_def(&mut self, params: &[Destructure], block: &Block<Captured>, capture: &Capture, pos: &Position) -> Type<C> {
    let (inds, types): (Vec<_>, Vec<_>) = capture
      .undeclared()
      .iter()
      .map(|u| {
        let (ind, tp) = self.resolve_local(u).with_context(|| format!("unable to resolve {}", u)).unwrap();
        (ind, (u.clone(), tp))
      })
      .unzip();

    let fn_def = self.pass.define(|| FnDef::new(params.to_vec(), block.clone(), types, pos.clone()));
    let const_ind = self.add_constant(Constant::FnDef(fn_def.inner()));

    self.emit(Opcode::Closure(const_ind, inds), pos.clone());
    Type::FnDef(fn_def)
  }

  async fn call(&mut self, expr: &ExpressionMeta<Captured>, args: &[ExpressionMeta<Captured>]) -> Type<C> {
    let pos = expr.pos();
    match self.expression(expr).await {
      Type::Unknown => Type::Unknown,
      Type::FnDef(f) => self.call_fn(f, args, pos).await,
      Type::Native(f) => self.call_fn(f, args, pos).await,
      t => panic!("Can't call an expression of type {:?}", t)
    }
  }

  async fn call_fn<T: Callable<C>>(
    &mut self, mut fn_def: CallableDef<T, C>, args: &[ExpressionMeta<Captured>], pos: &Position
  ) -> Type<C>
  where
    CallableDef<T, C>: fmt::Debug
  {
    let args: Vec<Type<C>> = fn_def.unpack_args(self, args).await;

    let ctype = self.pass_1(&mut fn_def, &args).await;

    if self.allowed_pass_2() {
      assert!(!ctype.is_unknown(), "Can't determine recursive type for {:?}.", fn_def);
      let ctype2 = self.pass_2(&mut fn_def, &args, pos).await;
      assert!(ctype.similar(&ctype2), "mismatch pass types: {:?}, {:?}", ctype, ctype2);

      // Don't set the type to the pass 2 type: use the pass 1 type instead: because we're reusing fn_defs
      // in different passes, the two types should be at least similar.
      //
      // ctype = ctype2;
    }

    ctype
  }

  async fn pass_1<T: Callable<C>>(&mut self, fn_def: &mut CallableDef<T, C>, args: &[Type<C>]) -> Type<C> {
    self.pass.incr_sub_pass();
    pass_1(fn_def, args, self.pass.sub_pass()).await
  }

  async fn pass_2<T: Callable<C>>(
    &mut self, fn_def: &mut CallableDef<T, C>, args: &[Type<C>], pos: &Position
  ) -> Type<C> {
    let (fn_ind, fn_type) = pass_2(fn_def, args, self.pass.sub_pass().two()).await;
    self.emit(Opcode::Call(fn_ind, args.len()), pos.clone());
    fn_type
  }

  fn allowed_pass_2(&self) -> bool { self.pass.allowed_pass_2() }
  fn begin_scope(&mut self) { self.scope.begin_scope() }
  fn end_scope(&mut self) { self.scope.end_scope() }
  fn reserve_used(&mut self) { self.scope.reserve_used(); }
  fn restore_used(&mut self) { self.scope.restore_used(); }
  fn mark_initialized(&mut self, neg_ind: usize, t: Type<C>) { self.scope.mark_initialized(neg_ind, t); }
  fn resolve_local(&mut self, name: &str) -> Option<(usize, Type<C>)> { self.scope.resolve_local(name) }
  fn declare_variable(&mut self, name: String) { self.scope.add_local(name); }

  pub fn emit(&mut self, op: Opcode, pos: Position) {
    if self.pass.is_emitting() {
      self.chunk.add_opcode(op, pos);
    }
  }

  pub fn add_constant(&mut self, c: Constant<C>) -> usize {
    if self.pass.is_emitting() {
      self.chunk.add_constant(c)
    } else {
      0
    }
  }

  fn emit_jump(&mut self, opcode: Opcode, pos: Position) -> usize {
    if self.pass.is_emitting() {
      self.emit(opcode, pos);
      self.chunk.code_len() - 1
    } else {
      0
    }
  }

  pub fn patch_jump(&mut self, offset: usize) {
    if self.pass.is_emitting() {
      let jump = self.chunk.code_len() - offset - 1;
      self.chunk.patch_jump(offset, jump);
    }
  }
}

#[derive(Clone, Debug)]
pub struct FnInd {
  index: usize,
  extracts: Vec<Extract>
}

impl FnInd {
  pub fn new(index: usize, extracts: Vec<Extract>) -> FnInd { FnInd { index, extracts } }
  pub fn script() -> FnInd { FnInd::new(0, Vec::new()) }
  pub fn index(&self) -> usize { self.index }
  pub fn extracts(&self) -> &[Extract] { &self.extracts }
}

/// A runtime specification for extracting variables from a destructure assignment.
#[derive(Clone, Debug)]
pub struct Extract {
  indexes: Vec<Vec<usize>>
}

impl Extract {
  pub fn solo() -> Extract { Extract { indexes: vec![Vec::new()] } }

  /// An solo extract doesn't need to be opcoded, because it's a bare identifier, which is already on the
  /// stack. See also `is_empty`.
  pub fn is_solo(&self) -> bool { self.indexes.len() == 1 && self.indexes[0].is_empty() }

  /// An empty extract needs to be opcoded to be removed from the stack.
  ///
  /// This would happen with e.g. an empty array destructure (`[] = [1, 2, 3]`), or a blank destructure `_`.
  /// There's nothing to extract, but the VM still needs to pop the source structure from the stack.
  pub fn is_empty(&self) -> bool { self.indexes.is_empty() }

  /// Extracts individual values from a source value, using the constructed indexes of this extract.
  pub fn extracted<C: Custom>(&self, mut val: Value<C>) -> impl Iterator<Item = Value<C>> + '_ {
    self.indexes.iter().map(move |index| {
      let mut target = &mut val;
      for ind in index {
        target = target.as_array_mut().get_mut(*ind).unwrap();
      }
      target.shift()
    })
  }

  /// Extracts individual types from a source type, using the constructed indexes of this extract.
  pub fn extracted_refs<'t, C: Custom>(&'t self, rtype: &'t Type<C>) -> impl Iterator<Item = &'t Type<C>> + 't {
    self.indexes.iter().map(move |index| {
      let mut target = rtype;
      for ind in index {
        target = match target {
          Type::Array(a) => a.get(*ind),
          Type::Object(o) => o.at(*ind),
          other => panic!("Not an indexable type: {:?}", other)
        };
      }
      target
    })
  }

  /// Extracts individual types from a source type, using the constructed indexes of this extract.
  pub fn extracted_types<C: Custom>(&self, mut rtype: Type<C>) -> impl Iterator<Item = Type<C>> + '_ {
    self.indexes.iter().map(move |index| {
      let mut target = &mut rtype;
      for ind in index {
        target = match target {
          Type::Array(a) => a.get_mut(*ind),
          Type::Object(o) => o.at_mut(*ind),
          other => panic!("Not an indexable type: {:?}", other)
        };
      }
      target.clone()
    })
  }
}

/// Something that visits a destructure. Default implementations simply traverse deeper.
pub trait DestructVisitor<C: Custom> {
  fn visit_destructure(&mut self, compiler: &mut Compiler<C>, dest: &Destructure) {
    match dest {
      Destructure::Identifier(s) => self.visit_identifier(compiler, s),
      Destructure::Array(a) => self.visit_array(compiler, a),
      Destructure::Object(o) => self.visit_object(compiler, o),
      Destructure::Blank => self.visit_blank(compiler)
    }
  }

  fn visit_blank(&mut self, _compiler: &mut Compiler<C>) {}

  fn visit_identifier(&mut self, _compiler: &mut Compiler<C>, _id: &str) {}

  fn visit_array(&mut self, compiler: &mut Compiler<C>, array: &[Destructure]) {
    for d in array {
      self.visit_destructure(compiler, d)
    }
  }

  fn visit_object(&mut self, compiler: &mut Compiler<C>, object: &[(String, Destructure)]) {
    for (_, d) in object {
      self.visit_destructure(compiler, d)
    }
  }
}

/// A chunk of bytecode.
///
/// The chunk includes: the constants found in the source; the actual instructions to run; and the source
/// position of each instruction.
#[derive(Clone, Debug)]
pub struct Chunk<C: Custom> {
  constants: Constants<C>,
  code: Vec<Opcode>,
  positions: Vec<Position>
}

impl<C: Custom> Default for Chunk<C> {
  fn default() -> Chunk<C> { Chunk::new() }
}

impl<C: Custom> Chunk<C> {
  pub fn new() -> Chunk<C> { Chunk { code: Vec::new(), constants: Constants::new(), positions: Vec::new() } }
  pub fn constants(&self) -> &Constants<C> { &self.constants }
  pub fn code(&self) -> &[Opcode] { &self.code }
  pub fn positions(&self) -> &[Position] { &self.positions }
  pub fn code_len(&self) -> usize { self.code.len() }

  pub fn at(&self, ip: usize) -> Option<&Opcode> { self.code.get(ip) }
  pub fn pos_at(&self, ip: usize) -> Option<&Position> { self.positions.get(ip) }
  pub fn constant_at(&self, ind: usize) -> Option<&Constant<C>> { self.constants.at(ind) }

  pub fn disassemble(&self, name: &str) {
    println!("Chunk \"{}\"", name);
    println!("  constants:");
    for (i, c) in self.constants.values.iter().enumerate() {
      println!("    {:>4} {:?}", i, c);
    }
    println!("  instructions:");
    for (i, (c, p)) in self.code.iter().zip(self.positions.iter()).enumerate() {
      println!("    {:>04} {:?} at {}", i, c, p);
    }
  }

  pub fn add_constant(&mut self, val: Constant<C>) -> usize {
    self.constants.push(val);
    self.constants.len() - 1
  }

  pub fn add_opcode(&mut self, op: Opcode, pos: Position) {
    self.code.push(op);
    self.positions.push(pos);
  }

  pub fn patch_jump(&mut self, ind: usize, val: usize) {
    match self.code.get_mut(ind) {
      Some(Opcode::Jump(v)) | Some(Opcode::JumpIfFalse(v)) => {
        *v = val;
      }
      other => panic!("Not a jump: {:?}", other)
    }
  }
}

#[derive(Clone, Debug)]
pub struct Constants<C: Custom> {
  values: Vec<Constant<C>>
}

impl<C: Custom> Default for Constants<C> {
  fn default() -> Constants<C> { Constants::new() }
}

impl<C: Custom> Constants<C> {
  pub fn new() -> Constants<C> { Constants { values: Vec::new() } }
  pub fn values(&self) -> &[Constant<C>] { &self.values }
  pub fn push(&mut self, val: Constant<C>) { self.values.push(val); }
  pub fn len(&self) -> usize { self.values.len() }
  pub fn is_empty(&self) -> bool { self.values.is_empty() }
  pub fn at(&self, ind: usize) -> Option<&Constant<C>> { self.values.get(ind) }
}

// TODO(later): use variable-length byte processing to compress the chunk.
/// The actual instructions available to our VM.
#[derive(Clone, Debug)]
pub enum Opcode {
  Constant(usize),
  Negate,
  Not,
  Add,
  Subtract,
  Multiply,
  Divide,
  Mod,
  And,
  Or,
  Gt,
  Gte,
  Lt,
  Lte,
  Equals,
  NotEquals,
  Return,
  GetLocal(usize),
  Array(usize),
  Object(Vec<usize>),
  Closure(usize, Vec<usize>),
  GetIndex(usize),
  GetJsonIndex(usize),
  GetJsonKey(String),
  Extract(Extract),
  JumpIfFalse(usize),
  Jump(usize),
  Pop,
  Call(usize, usize) // morph_index, destructed_argc
}

impl Opcode {
  pub fn initial_jump_if_false() -> Opcode { Self::JumpIfFalse(0) }
  pub fn initial_jump() -> Opcode { Self::Jump(0) }
}

#[derive(Debug)]
pub struct Pass<C: Custom> {
  pass_type: PassType,
  fn_defs: Vec<FnDef<C>>,
  index: usize,
  ntv_defs: Vec<NativeDef<C>>,
  ntv_index: usize,
  passes: Vec<Pass<C>>,
  pass_index: usize
}

impl<C: Custom> Pass<C> {
  pub fn one() -> Pass<C> {
    Pass {
      pass_type: PassType::One,
      fn_defs: Vec::new(),
      index: 0,
      ntv_defs: Vec::new(),
      ntv_index: 0,
      passes: Vec::new(),
      pass_index: 0
    }
  }

  pub fn two(&mut self) -> &mut Self {
    assert!(self.is_one());
    self.pass_type = PassType::Two;
    self.start();
    self
  }

  pub fn is_one(&self) -> bool { matches!(self.pass_type, PassType::One) }
  pub fn is_two(&self) -> bool { matches!(self.pass_type, PassType::Two) }
  pub fn is_emitting(&self) -> bool { self.is_two() }
  pub fn allowed_pass_2(&self) -> bool { self.is_two() }
  pub fn pass_type(&self) -> &PassType { &self.pass_type }

  pub fn start(&mut self) {
    self.index = 0;
    self.ntv_index = 0;
    self.pass_index = 0;
  }

  pub fn done_status(&self, done_type: &Type<C>) -> ImplStatus<C> {
    match self.pass_type {
      PassType::One => ImplStatus::Discovered(done_type.clone()),
      PassType::Two => ImplStatus::Completed(done_type.clone())
    }
  }

  pub fn define<F: FnOnce() -> FnDef<C>>(&mut self, f: F) -> FnDef<C> {
    // TODO(later, shortcut): we assume here that every pass hits function definitions in the same code in the
    // same order. This is valid now, but it may not always be: for example, if we implement constexpr shortcuts
    // or some other refinement that allows us to conditionally skip code compilation. We will need a better
    // method of identification in that case (position-based indexing or something like that).
    //
    // This also applies to native defs (`define_native`) and passes (`incr_sub_pass`).

    assert!(
      self.fn_defs.len() > self.index || (self.fn_defs.len() == self.index && self.is_one()),
      "no fn_def {}: len={} for {:?}",
      self.index,
      self.fn_defs.len(),
      self.pass_type
    );

    if self.fn_defs.len() == self.index {
      self.fn_defs.push(f());
    } else if self.fn_defs[self.index].is_unknown() {
      self.fn_defs[self.index] = f();
    }

    // TODO: verify that the constructed native_def has the same something as the cached native_def.

    self.index += 1;
    self.fn_defs[self.index - 1].clone()
  }

  pub fn define_native<F: FnOnce() -> NativeDef<C>>(&mut self, f: F) -> NativeDef<C> {
    assert!(
      self.ntv_defs.len() > self.ntv_index || (self.ntv_defs.len() == self.ntv_index && self.is_one()),
      "no native {}: len={} for {:?}",
      self.ntv_index,
      self.ntv_defs.len(),
      self.pass_type
    );

    if self.ntv_defs.len() == self.ntv_index {
      self.ntv_defs.push(f());
    } else if self.ntv_defs[self.ntv_index].is_unknown() {
      self.ntv_defs[self.ntv_index] = f();
    }

    // TODO: verify that the constructed fn_def has the same position as the cached fn_def.

    self.ntv_index += 1;
    self.ntv_defs[self.ntv_index - 1].clone()
  }

  pub fn incr_sub_pass(&mut self) {
    self.pass_index += 1;
    let pass_index = self.pass_index - 1;

    assert!(
      self.passes.len() > pass_index || (self.passes.len() == pass_index && self.is_one()),
      "no pass {}: len={} for {:?}",
      pass_index,
      self.passes.len(),
      self.pass_type
    );

    assert!(self.passes.len() > pass_index || (self.passes.len() == pass_index && self.is_one()));

    if self.passes.len() == pass_index {
      self.passes.push(Pass::one());
    }
  }

  pub fn sub_pass(&mut self) -> &mut Pass<C> {
    let pass_index = self.pass_index - 1;
    self.passes[pass_index].start();
    &mut self.passes[pass_index]
  }
}

#[derive(Debug)]
pub enum PassType {
  One,
  Two
}

pub fn add_natives<C: Custom>(
  script: &mut FnDef<C>, natives: &[(NativeBuildFn<C>, NativeRunFn<C>, &str)]
) -> Vec<Value<C>> {
  // "globals" are just top-level items added to context, as if they were captured variables.

  let (cap_types, vals): (Vec<_>, Vec<_>) = natives
    .iter()
    .map(|(bfn, rfn, name)| {
      let def = NativeDef::new(*bfn, *rfn);
      ((name.to_string(), Type::Native(def.clone())), Value::NativeDef(def.inner(), Vec::new(), Default::default()))
    })
    .unzip();

  script.set_captures(cap_types);
  vals
}

pub async fn double_build<C: Custom>(f: &Type<C>, args: Vec<Type<C>>) -> (FnInd, Type<C>) {
  let mut one = Pass::one();
  let _ = build(f, args.clone(), &mut one).await;
  build(f, args, one.two()).await
}

pub async fn build<C: Custom>(f: &Type<C>, args: Vec<Type<C>>, pass: &mut Pass<C>) -> (FnInd, Type<C>) {
  match f {
    Type::FnDef(f) => build_def(f, args, pass).await,
    Type::Native(f) => build_def(f, args, pass).await,
    other => panic!("Not a callable type: {:?}", other)
  }
}

pub async fn build_def<C: Custom, T: Callable<C>>(
  def: &CallableDef<T, C>, args: Vec<Type<C>>, pass: &mut Pass<C>
) -> (FnInd, Type<C>) {
  let (extracts, args) = T::extracts_and_args(def, &args);
  let (inst_ind, ftype) = build_with_at(def, &args, pass).await;
  (FnInd::new(inst_ind, extracts), ftype)
}

async fn build_with_at<C: Custom, T: Callable<C>>(
  fn_def: &CallableDef<T, C>, args: &[Type<C>], pass: &mut Pass<C>
) -> (usize, Type<C>) {
  match pass.pass_type() {
    PassType::One => (0, pass_1(fn_def, args, pass).await),
    PassType::Two => pass_2(fn_def, args, pass).await
  }
}

async fn pass_1<C: Custom, T: Callable<C>>(
  fn_def: &CallableDef<T, C>, args: &[Type<C>], pass: &mut Pass<C>
) -> Type<C> {
  // footnote: (3)

  // Briefly borrow def to find the status.
  //
  // TODO(later): can we instead borrow the status and drop (or clone the type) shortly afterwards instead?
  // That way, `ImplStatus` wouldn't have to be `Clone`, which better reflects how it should be used (the
  // status should only exist with the impl).
  let (pending, (_, status)) = fn_def.find_status(args);

  pass.incr_sub_pass();

  if pending {
    // footnote: (2)
    if let Some(t) = status.status_type() {
      t
    } else {
      Type::recursion()
    }
  } else {
    // footnote: (1)
    if let Some(t) = status.status_type() {
      if !t.is_unknown() {
        return t;
      }
    }

    fn_def.compile(args, pass).await.1
  }
}

async fn pass_2<C: Custom, T: Callable<C>>(
  fn_def: &CallableDef<T, C>, args: &[Type<C>], pass: &mut Pass<C>
) -> (usize, Type<C>) {
  // footnote: (3)

  // Briefly borrow def to find the status.
  let (pending, (morph_ind, status)) = fn_def.find_status(args);

  if pending {
    // footnote: (2)
    if let Some(t) = status.status_type() {
      assert!(status.is_completed(), "We should have a completed type for recursive pass 2.");
      (morph_ind, t)
    } else {
      panic!("We should have a type by pass 2");
    }
  } else {
    if let Some(t) = status.status_type() {
      if status.is_completed() {
        return (morph_ind, t);
      }
    }

    fn_def.compile(args, pass).await
  }
}

pub fn find_extracts<C: Custom>(fn_def: &FnDef<C>, args: &[Type<C>]) -> Vec<Extract> {
  assert_eq!(args.len(), fn_def.params().len());
  fn_def
    .params()
    .iter()
    .zip(args.iter())
    .map(|(p, a)| {
      // TODO(later): we shouldn't need to construct a whole compiler just to get an extract.
      Compiler::new(&mut Pass::one()).find_extract(p, a, false)
    })
    .collect()
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::enhancer::enhance;
  use crate::parser::parse;
  use crate::test::setup;

  #[tokio::test]
  #[should_panic]
  async fn single_use() {
    // This only works with the single-use hack for literal "SINGLE"

    setup();
    let content = r#"a = "SINGLE"; b = a; c = a;"#;

    let script = parse(content).await;
    let enhanced = enhance(script);
    let _: FnDef<()> = compile(enhanced).await;
  }
}
