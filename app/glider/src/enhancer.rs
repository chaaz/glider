//! A compiation pass that is used to add additional information to an AST.
//!
//! Currently, the only enhancement made is generating variable capture information at the function level to
//! generate the correct closure bytecode.

use crate::parser::{Assignment, BinaryOp, Block, Destructure, DotIndex, Enhancement, Expression, ExpressionMeta,
                    Scope, Script, UnaryOp};
use crate::scanner::Position;
use crate::value::Literal;

pub fn enhance(script: Script<()>) -> Script<Captured> {
  let mut enhancer = Enhancer::new();
  Script::new(script.source().to_string(), enhancer.block(script.block()))
}

#[derive(Debug, Clone)]
pub struct Captured {}

impl Enhancement for Captured {
  type Function = Capture;
}

#[derive(Debug, Clone)]
pub struct Capture {
  undeclared: Vec<String>
}

impl Default for Capture {
  fn default() -> Capture { Capture::new() }
}

impl Capture {
  pub fn new() -> Capture { Capture { undeclared: Vec::new() } }

  /// The variables used by a function, but not declared in the body or parameters.
  pub fn undeclared(&self) -> &[String] { &self.undeclared }

  pub fn add_undeclared(&mut self, v: String) {
    if !self.undeclared.contains(&v) {
      self.undeclared.push(v);
    }
  }
}

pub struct Enhancer {
  captured: Capture,
  scope: Scope<()>
}

impl Default for Enhancer {
  fn default() -> Enhancer { Enhancer::new() }
}

impl Enhancer {
  pub fn new() -> Enhancer { Enhancer { captured: Capture::new(), scope: Scope::new() } }

  pub fn captured(&self) -> &Capture { &self.captured }
  pub fn into_capture(self) -> Capture { self.captured }

  pub fn function(&mut self, params: &[Destructure], block: &Block<()>) -> Block<Captured> {
    self.begin_scope();
    for v in params.iter().flat_map(|d| d.identifiers()) {
      self.declare_variable(v.to_string());
      self.mark_initialized(1);
    }

    let result = self.block(block);
    self.end_scope();
    result
  }

  pub fn block(&mut self, block: &Block<()>) -> Block<Captured> {
    self.begin_scope();

    let asgns = block.assignments().iter().map(|a| self.assignment(a)).collect();
    let value = self.expression(block.value());

    self.end_scope();
    Block::new(asgns, value)
  }

  pub fn assignment(&mut self, asgn: &Assignment<()>) -> Assignment<Captured> {
    self.declare_all_variables(asgn.lhs());
    let rhs = self.expression(asgn.rhs());
    self.mark_all_initialized(asgn.lhs());

    Assignment::new(asgn.pos().clone(), asgn.lhs().clone(), rhs)
  }

  pub fn declare_all_variables(&mut self, dstr: &Destructure) {
    struct Declare;
    impl DestructVisitor for Declare {
      fn visit_identifier(&mut self, compiler: &mut Enhancer, id: &str) { compiler.declare_variable(id.to_string()); }
    }
    Declare.visit_destructure(self, dstr);
  }

  pub fn mark_all_initialized(&mut self, lhs: &Destructure) {
    struct Marker {
      init_mark: usize
    }
    impl DestructVisitor for Marker {
      fn visit_identifier(&mut self, compiler: &mut Enhancer, _id: &str) {
        compiler.mark_initialized(self.init_mark);
        self.init_mark -= 1;
      }
    }

    Marker { init_mark: lhs.identifier_count() }.visit_destructure(self, lhs);
  }

  pub fn expression(&mut self, expr: &ExpressionMeta<()>) -> ExpressionMeta<Captured> {
    match expr.expr() {
      Expression::Variable(name) => self.variable(name, expr.pos()),
      Expression::Literal(v) => self.literal(v, expr.pos()),
      Expression::Unary(op, expr) => self.unary(*op, expr),
      Expression::Binary(op, lexpr, rexpr) => self.binary(*op, lexpr, rexpr),
      Expression::Array(exprs) => self.array(exprs, expr.pos()),
      Expression::Object(pairs) => self.object(pairs, expr.pos()),
      Expression::Dot(expr, di) => self.dot(expr, di),
      Expression::Index(expr, i) => self.index(expr, *i),
      Expression::IfBlock(cond, block, elseifs, last) => self.if_block(cond, block, elseifs, last, expr.pos()),
      Expression::And(expr1, expr2) => self.and(expr1, expr2),
      Expression::Or(expr1, expr2) => self.or(expr1, expr2),
      Expression::Function(params, block, _) => self.fn_def(params, block, expr.pos()),
      Expression::Call(expr, args) => self.call(expr, args)
    }
  }

  fn variable(&mut self, name: &str, pos: &Position) -> ExpressionMeta<Captured> {
    if self.resolve_local(name).is_none() {
      self.captured.add_undeclared(name.to_string());
    }

    ExpressionMeta::new(Expression::Variable(name.to_string()), pos.clone())
  }

  fn literal(&mut self, lit: &Literal, pos: &Position) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(Expression::Literal(lit.clone()), pos.clone())
  }

  fn unary(&mut self, op: UnaryOp, expr: &ExpressionMeta<()>) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(Expression::Unary(op, Box::new(self.expression(expr))), expr.pos().clone())
  }

  fn binary(
    &mut self, op: BinaryOp, lexpr: &ExpressionMeta<()>, rexpr: &ExpressionMeta<()>
  ) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(
      Expression::Binary(op, Box::new(self.expression(lexpr)), Box::new(self.expression(rexpr))),
      lexpr.pos().clone()
    )
  }

  fn array(&mut self, exprs: &[ExpressionMeta<()>], pos: &Position) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(Expression::Array(exprs.iter().map(|e| self.expression(e)).collect()), pos.clone())
  }

  fn object(&mut self, pairs: &[(String, ExpressionMeta<()>)], pos: &Position) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(
      Expression::Object(pairs.iter().map(|(k, e)| (k.clone(), self.expression(e))).collect()),
      pos.clone()
    )
  }

  fn dot(&mut self, expr: &ExpressionMeta<()>, di: &DotIndex) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(Expression::Dot(Box::new(self.expression(expr)), di.clone()), expr.pos().clone())
  }

  fn index(&mut self, expr: &ExpressionMeta<()>, i: u64) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(Expression::Index(Box::new(self.expression(expr)), i), expr.pos().clone())
  }

  fn if_block(
    &mut self, cond: &ExpressionMeta<()>, block: &Block<()>, elseifs: &[(ExpressionMeta<()>, Block<()>)],
    last: &Block<()>, pos: &Position
  ) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(
      Expression::IfBlock(
        Box::new(self.expression(cond)),
        self.block(block),
        elseifs.iter().map(|(e, b)| (self.expression(e), self.block(b))).collect(),
        self.block(last)
      ),
      pos.clone()
    )
  }

  fn and(&mut self, left: &ExpressionMeta<()>, right: &ExpressionMeta<()>) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(
      Expression::And(Box::new(self.expression(left)), Box::new(self.expression(right))),
      left.pos().clone()
    )
  }

  fn or(&mut self, left: &ExpressionMeta<()>, right: &ExpressionMeta<()>) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(
      Expression::Or(Box::new(self.expression(left)), Box::new(self.expression(right))),
      left.pos().clone()
    )
  }

  fn fn_def(&mut self, params: &[Destructure], block: &Block<()>, pos: &Position) -> ExpressionMeta<Captured> {
    let mut enhancer = Enhancer::new();
    let enhanced_block = enhancer.function(params, block);
    let capture = enhancer.into_capture();

    // Captured variables must be available in the surrounding function, even if that function has to also
    // capture them.
    for name in capture.undeclared() {
      if self.resolve_local(name).is_none() {
        self.captured.add_undeclared(name.to_string());
      }
    }

    ExpressionMeta::new(Expression::Function(params.to_vec(), enhanced_block, capture), pos.clone())
  }

  fn call(&mut self, expr: &ExpressionMeta<()>, args: &[ExpressionMeta<()>]) -> ExpressionMeta<Captured> {
    ExpressionMeta::new(
      Expression::Call(Box::new(self.expression(expr)), args.iter().map(|a| self.expression(a)).collect()),
      expr.pos().clone()
    )
  }

  fn begin_scope(&mut self) { self.scope.begin_scope() }
  fn end_scope(&mut self) { self.scope.end_scope() }
  fn mark_initialized(&mut self, neg_ind: usize) { self.scope.mark_initialized(neg_ind, ()); }
  fn resolve_local(&mut self, name: &str) -> Option<usize> { self.scope.resolve_local(name).map(|a| a.0) }
  fn declare_variable(&mut self, name: String) { self.scope.add_local(name); }
}

/// Something that visits a destructure. Default implementations simply traverse deeper.
pub trait DestructVisitor {
  fn visit_destructure(&mut self, compiler: &mut Enhancer, dest: &Destructure) {
    match dest {
      Destructure::Identifier(s) => self.visit_identifier(compiler, s),
      Destructure::Array(a) => self.visit_array(compiler, a),
      Destructure::Object(o) => self.visit_object(compiler, o),
      Destructure::Blank => ()
    }
  }

  fn visit_identifier(&mut self, _compiler: &mut Enhancer, _id: &str) {}

  fn visit_array(&mut self, compiler: &mut Enhancer, array: &[Destructure]) {
    for d in array {
      self.visit_destructure(compiler, d)
    }
  }

  fn visit_object(&mut self, compiler: &mut Enhancer, object: &[(String, Destructure)]) {
    for (_, d) in object {
      self.visit_destructure(compiler, d)
    }
  }
}
