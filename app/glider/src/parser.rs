//! The parser (1st pass) for Glider, that constructs an AST from source code.
//!
//! This is a hand-crafted parser that uses a simple top-down algorithm (include Pratt's "top-down operator
//! precedence parsing") to generate a no-surprises AST.

use crate::pick_opt;
use crate::scanner::{Position, Scanner, Token, TokenMeta, TokenType};
use crate::value::Literal;
use anyhow::Error;
use futures::future::Future;
use lazy_static::lazy_static;
use std::cmp::max;
use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::str::FromStr;
use std::vec::IntoIter;
use tracing::debug;

pub async fn parse(source: &str) -> Script<()> {
  let mut scanner = Scanner::new(source);
  let mut parser = Parser::new(scanner.drain_into_iter());
  Script::new(source.to_string(), parser.parse().await)
}

const MAX_LOCALS: usize = 255;

lazy_static! {
  pub(crate) static ref RULES: HashMap<TokenType, Rule> = construct_rules();
}

type CFut<'r, T> = Pin<Box<dyn Future<Output = T> + Send + 'r>>;
type TFut<'r> = Pin<Box<dyn Future<Output = ExpressionMeta<()>> + Send + 'r>>;
type Prefix = Option<for<'r> fn(&'r mut Parser) -> TFut<'r>>;
type Infix = Option<for<'r> fn(&'r mut Parser, ExpressionMeta<()>) -> TFut<'r>>;

pub struct Parser {
  scanner: IntoIter<TokenMeta>,
  current: TokenMeta,
  previous: TokenMeta,
  last_pos: Position,
  had_error: bool,
  panic_mode: bool
}

impl Parser {
  pub fn new(scanner: IntoIter<TokenMeta>) -> Parser {
    let mut parser = Parser {
      scanner,
      current: TokenMeta::new(Token::Bof, Position::zero()),
      previous: TokenMeta::new(Token::Bof, Position::zero()),
      had_error: false,
      panic_mode: false,
      last_pos: Position::zero()
    };
    parser.advance();
    parser
  }

  pub async fn parse(&mut self) -> Block<()> { self.body(false).await }

  /// Parse a block's content and curly braces.
  pub async fn block(&mut self) -> Block<()> {
    self.consume(TokenType::OpenCurl);
    let block = self.body(true).await;
    self.consume(TokenType::CloseCurl);
    block
  }

  /// Parse a block's content.
  pub async fn body(&mut self, end_curly: bool) -> Block<()> {
    let mut assigns = Vec::new();

    loop {
      let tt = self.current_tt();
      if tt == TokenType::CloseCurl || tt == TokenType::Eof {
        // We're at the of the block with no trailing expression. Create a block that returns `Unit`.
        if end_curly != (tt == TokenType::CloseCurl) {
          panic!("Unexpected block end: {:?}", tt)
        }
        return Block::new(assigns, meta(Expression::Literal(Literal::Unit), self.current_pos()));
      }

      // Not at the end, the next item should be an expression (or expression-like) construction.
      let expr = self.expression().await;

      match self.current_tt() {
        tt @ TokenType::CloseCurl | tt @ TokenType::Eof => {
          // We're at the end of the block, the expression is the block value. We expect the caller to consume
          // the ending token.
          if end_curly != (tt == TokenType::CloseCurl) {
            panic!("Unexpected block end {:?} at {}", tt, self.current_pos())
          }
          self.maybe_synchronize();
          return Block::new(assigns, expr);
        }
        TokenType::Equals => {
          self.advance();

          // Whoops, that was actually the left-hand of an assignment.
          let (lhs, pos) = expr.into_destructure();
          let rhs = self.expression().await;

          if !rhs.allowed_skip_semi() || self.token_check(TokenType::Semi) {
            self.consume(TokenType::Semi);
          }

          assigns.push(Assignment::new(pos, lhs, rhs));
          self.maybe_synchronize();
        }
        TokenType::Semi => {
          // We have an expression statement.
          self.advance();
          assigns.push(Assignment::new(expr.pos().clone(), Destructure::Blank, expr));
          self.maybe_synchronize();
        }
        tt => panic!("Unknown token following expression {:?} : {:?}", expr, tt)
      }
    }
  }

  pub async fn expression(&mut self) -> ExpressionMeta<()> { self.parse_precedence(Precedence::Or).await }

  pub async fn variable(&mut self) -> ExpressionMeta<()> {
    if let Token::Identifier(s) = self.previous.token() {
      let name = s.to_string();
      meta(Expression::Variable(name), self.previous_pos())
    } else {
      panic!("Unexpected token for named variable {:?}", self.previous)
    }
  }

  async fn unary(&mut self) -> ExpressionMeta<()> {
    let td = self.previous_tt();
    let pos = self.previous_pos();
    let base = self.parse_precedence(Precedence::Unary).await;

    match td {
      TokenType::Minus => meta(Expression::Unary(UnaryOp::Negate, Box::new(base)), pos),
      TokenType::Bang => meta(Expression::Unary(UnaryOp::Not, Box::new(base)), pos),
      other => panic!("Unexpected unary op: {:?}", other)
    }
  }

  async fn literal(&mut self) -> ExpressionMeta<()> {
    let pos = self.previous_pos();
    match self.previous.token() {
      Token::Unit => meta(Expression::Literal(Literal::Unit), pos),
      Token::IntLit(v) => meta(Expression::Literal(to_value::<i64>(v)), pos),
      Token::FloatLit(v) => meta(Expression::Literal(to_value::<f64>(v)), pos),
      Token::TrueLit => meta(Expression::Literal(Literal::Bool(true)), pos),
      Token::FalseLit => meta(Expression::Literal(Literal::Bool(false)), pos),
      Token::StringLit(v) => meta(Expression::Literal(to_value::<String>(v)), pos),
      other => panic!("Unexpected literal {:?}", other)
    }
  }

  async fn array(&mut self) -> ExpressionMeta<()> {
    let pos = self.current_pos();

    struct ExprBuild;
    impl CommaElmBuild for ExprBuild {
      type Element = ExpressionMeta<()>;
      fn build_element(parser: &mut Parser) -> CFut<ExpressionMeta<()>> { Box::pin(parser.expression()) }
    }
    let exprs = self.handle_commas::<ExprBuild>(TokenType::CloseSquare, 255, "array element").await;

    meta(Expression::Array(exprs), pos)
  }

  async fn object(&mut self) -> ExpressionMeta<()> {
    let pos = self.current_pos();

    struct PairBuild;
    impl CommaElmBuild for PairBuild {
      type Element = (String, ExpressionMeta<()>);
      fn build_element(parser: &mut Parser) -> CFut<(String, ExpressionMeta<()>)> { Box::pin(parser.object_pair()) }
    }
    let pairs = self.handle_commas::<PairBuild>(TokenType::CloseCurl, 255, "object pair").await;

    meta(Expression::Object(pairs), pos)
  }

  async fn object_pair(&mut self) -> (String, ExpressionMeta<()>) {
    self.consume(TokenType::Identifier);

    if let Token::Identifier(s) = self.previous.token() {
      let s = s.clone();
      if self.current_tt() == TokenType::Colon {
        self.advance();
        (s, self.expression().await)
      } else {
        (s, self.variable().await)
      }
    } else {
      panic!("object pair must start with an identifier")
    }
  }

  async fn grouping(&mut self) -> ExpressionMeta<()> {
    let pos = self.current_pos();
    let base = self.expression().await;
    self.consume(TokenType::CloseParen);
    meta(base.into_expr(), pos)
  }

  async fn if_block(&mut self) -> ExpressionMeta<()> {
    let pos = self.current_pos();

    let init_test = self.expression().await;
    let init_block = self.block().await;

    let mut elseifs = Vec::new();
    while let TokenType::ElseifWord = self.current_tt() {
      self.advance();

      let elseif_test = self.expression().await;
      let elseif_block = self.block().await;
      elseifs.push((elseif_test, elseif_block));
    }

    self.consume(TokenType::ElseWord);
    let else_block = self.block().await;

    meta(Expression::IfBlock(Box::new(init_test), init_block, elseifs, else_block), pos)
  }

  async fn function(&mut self) -> ExpressionMeta<()> {
    let pos = self.previous_pos();

    let params = if self.current_tt() == TokenType::Unit {
      self.advance();
      Vec::new()
    } else {
      self.consume(TokenType::OpenParen);
      struct DestrBuild;
      impl CommaElmBuild for DestrBuild {
        type Element = Destructure;
        fn build_element(parser: &mut Parser) -> CFut<Destructure> { Box::pin(parser.destructure()) }
      }
      self.handle_commas::<DestrBuild>(TokenType::CloseParen, 255, "parameter").await
    };

    self.consume(TokenType::OpenCurl);
    let body = self.body(true).await;
    self.consume(TokenType::CloseCurl);

    meta(Expression::Function(params, body, ()), pos)
  }

  async fn binary(&mut self, lexpr: ExpressionMeta<()>) -> ExpressionMeta<()> {
    let pos = self.previous_pos();

    let tt = self.previous_tt();
    let precedence = get_rule(tt).precedence().up();
    let rexpr = self.parse_precedence(precedence).await;

    let op = match tt {
      TokenType::Plus => BinaryOp::Add,
      TokenType::Minus => BinaryOp::Subtract,
      TokenType::Star => BinaryOp::Multiply,
      TokenType::Slash => BinaryOp::Divide,
      TokenType::Percent => BinaryOp::Mod,
      TokenType::Gt => BinaryOp::Gt,
      TokenType::Lt => BinaryOp::Lt,
      TokenType::Gte => BinaryOp::Gte,
      TokenType::Lte => BinaryOp::Lte,
      TokenType::DoubleEq => BinaryOp::Equals,
      TokenType::NotEq => BinaryOp::NotEquals,
      other => panic!("Unexpected binary op: {:?}", other)
    };

    meta(Expression::Binary(op, Box::new(lexpr), Box::new(rexpr)), pos)
  }

  async fn index(&mut self, expr: ExpressionMeta<()>) -> ExpressionMeta<()> {
    let pos = self.current_pos();

    self.advance();
    match self.previous.token() {
      Token::IntLit(s) => {
        let ind = s.parse().unwrap();
        self.consume(TokenType::CloseSquare);
        meta(Expression::Index(Box::new(expr), ind), pos)
      }
      other => panic!("Index must be a literal integer, not {:?}.", other)
    }
  }

  async fn dot(&mut self, lexpr: ExpressionMeta<()>) -> ExpressionMeta<()> {
    let pos = self.current_pos();

    self.advance();
    match self.previous.token() {
      Token::Identifier(s) => meta(Expression::Dot(Box::new(lexpr), DotIndex::Identifier(s.clone())), pos),
      Token::IntLit(s) => meta(Expression::Dot(Box::new(lexpr), DotIndex::Integer(s.parse().unwrap())), pos),
      _ => panic!("Can't use {:?} as dot index.", self.previous_tt())
    }
  }

  async fn and(&mut self, lexpr: ExpressionMeta<()>) -> ExpressionMeta<()> {
    let pos = self.previous_pos();
    let rexpr = self.parse_precedence(Precedence::And).await;
    meta(Expression::And(Box::new(lexpr), Box::new(rexpr)), pos)
  }

  async fn or(&mut self, lexpr: ExpressionMeta<()>) -> ExpressionMeta<()> {
    let pos = self.previous_pos();
    let rexpr = self.parse_precedence(Precedence::Or).await;
    meta(Expression::Or(Box::new(lexpr), Box::new(rexpr)), pos)
  }

  async fn call(&mut self, lexpr: ExpressionMeta<()>) -> ExpressionMeta<()> {
    let pos = self.previous_pos();

    struct ExprBuild;
    impl CommaElmBuild for ExprBuild {
      type Element = ExpressionMeta<()>;
      fn build_element(parser: &mut Parser) -> CFut<ExpressionMeta<()>> { Box::pin(parser.expression()) }
    }
    let args = self.handle_commas::<ExprBuild>(TokenType::CloseParen, 255, "parameter").await;
    meta(Expression::Call(Box::new(lexpr), args), pos)
  }

  async fn call_unit(&mut self, lexpr: ExpressionMeta<()>) -> ExpressionMeta<()> {
    let pos = self.previous_pos();
    meta(Expression::Call(Box::new(lexpr), Vec::new()), pos)
  }

  async fn destructure(&mut self) -> Destructure {
    self.advance();

    match self.previous.token() {
      Token::Identifier(s) => {
        if s == "_" {
          Destructure::Blank
        } else {
          Destructure::Identifier(s.clone())
        }
      }
      Token::OpenSquare => {
        struct DestrBuild;
        impl CommaElmBuild for DestrBuild {
          type Element = Destructure;
          fn build_element(parser: &mut Parser) -> CFut<Destructure> { Box::pin(parser.destructure()) }
        }
        let dstrs = self.handle_commas::<DestrBuild>(TokenType::CloseSquare, 255, "array destructure").await;

        Destructure::Array(dstrs)
      }
      Token::OpenCurl => {
        struct DestrBuild;
        impl CommaElmBuild for DestrBuild {
          type Element = (String, Destructure);
          fn build_element(parser: &mut Parser) -> CFut<(String, Destructure)> { Box::pin(parser.destructure_pair()) }
        }
        let pairs = self.handle_commas::<DestrBuild>(TokenType::CloseCurl, 255, "array destructure").await;

        Destructure::Object(pairs)
      }
      other => panic!("Unexpected token to extract: {:?}", other)
    }
  }

  async fn destructure_pair(&mut self) -> (String, Destructure) {
    self.consume(TokenType::Identifier);

    if let Token::Identifier(s) = self.previous.token() {
      let s = s.clone();
      if self.current_tt() == TokenType::Colon {
        self.advance();
        (s, self.destructure().await)
      } else {
        (s.clone(), if s == "_" { Destructure::Blank } else { Destructure::Identifier(s) })
      }
    } else {
      panic!("destructure pair must start with an identifier")
    }
  }

  async fn parse_precedence(&mut self, prec: Precedence) -> ExpressionMeta<()> {
    self.advance();

    let prefix = self.previous_rule().prefix().expect("Expected expression.");
    let mut expr = prefix(self).await;

    while prec <= self.current_rule().precedence() {
      self.advance();
      let infix = self.previous_rule().infix().unwrap_or_else(|| panic!("No infix rule for {:?}", self.previous_tt()));
      expr = infix(self, expr).await;
    }

    expr
  }

  async fn handle_commas<B: CommaElmBuild>(
    &mut self, end: TokenType, max_elm: usize, elm_desc: &str
  ) -> Vec<B::Element> {
    let mut separated = true;
    let mut items = 0;
    let mut vec = Vec::new();

    while self.current_tt() != end {
      items += 1;
      if items > max_elm {
        panic!("More than {} {}.", max_elm, elm_desc);
      }
      if !separated {
        panic!("Missing comma after {}.", elm_desc);
      }

      vec.push(B::build_element(self).await);
      separated = false;

      if self.current_tt() == TokenType::Comma {
        self.consume(TokenType::Comma);
        separated = true;
      }
    }
    self.consume(end);

    vec
  }

  fn advance(&mut self) {
    let next = self.next_token();
    debug!("Got token: {:?}", next);
    self.previous = std::mem::replace(&mut self.current, next);
  }

  fn consume(&mut self, expect: TokenType) {
    if !self.token_match(expect) {
      self.error_current(&format!("Expected {:?}", expect));
    }
  }

  fn token_match(&mut self, ttd: TokenType) -> bool {
    if self.token_check(ttd) {
      self.advance();
      true
    } else {
      false
    }
  }

  fn next_token(&mut self) -> TokenMeta {
    for token in &mut self.scanner {
      if token.is_error() {
        error_token(&mut self.had_error, &mut self.panic_mode, &token, "parse error");
      } else {
        self.last_pos = token.pos().clone();
        return token;
      }
    }
    TokenMeta::new(Token::Eof, self.last_pos.clone())
  }

  fn maybe_synchronize(&mut self) {
    if self.panic_mode {
      self.synchronize();
    }
  }

  fn synchronize(&mut self) {
    // TODO: this should be done better
    self.panic_mode = false;

    while self.current_tt() != TokenType::Eof {
      if self.previous_tt() == TokenType::Semi {
        return;
      }

      match self.current_tt() {
        TokenType::Identifier | TokenType::OpenSquare | TokenType::OpenCurl => return,
        _ => ()
      }

      self.advance();
    }
  }

  fn current_rule(&self) -> &'static Rule { get_rule(self.current_tt()) }
  fn previous_rule(&self) -> &'static Rule { get_rule(self.previous_tt()) }
  fn previous_pos(&self) -> Position { self.previous.pos().clone() }
  fn current_pos(&self) -> Position { self.current.pos().clone() }
  fn current_tt(&self) -> TokenType { self.current.token().tt() }
  fn previous_tt(&self) -> TokenType { self.previous.token().tt() }
  fn token_check(&self, td: TokenType) -> bool { self.current_tt() == td }
  fn error_current(&mut self, msg: &str) { error_token(&mut self.had_error, &mut self.panic_mode, &self.current, msg); }
}

fn meta(expr: Expression<()>, pos: Position) -> ExpressionMeta<()> { ExpressionMeta::new(expr, pos) }

trait CommaElmBuild {
  type Element;
  fn build_element(parser: &mut Parser) -> CFut<Self::Element>;
}

fn get_rule(tt: TokenType) -> &'static Rule { &RULES[&tt] }

fn error_token(had_error: &mut bool, panic_mode: &mut bool, token: &TokenMeta, msg: &str) {
  *had_error = true;
  if *panic_mode {
    return;
  }
  *panic_mode = true;
  println!("Parse error: {:?}: {}", token, msg);
}

#[derive(Debug)]
pub struct Script<N: Enhancement> {
  source: String,
  block: Block<N>
}

impl<N: Enhancement> Script<N> {
  pub fn new(source: String, block: Block<N>) -> Script<N> { Script { source, block } }
  pub fn source(&self) -> &str { &self.source }
  pub fn block(&self) -> &Block<N> { &self.block }
  pub fn into_block(self) -> Block<N> { self.block }
}

#[derive(Clone, Debug)]
pub struct Block<N: Enhancement> {
  assignments: Vec<Assignment<N>>,
  value: Box<ExpressionMeta<N>>
}

impl<N: Enhancement> Block<N> {
  pub fn new(assignments: Vec<Assignment<N>>, expr: ExpressionMeta<N>) -> Block<N> {
    Block { assignments, value: Box::new(expr) }
  }

  pub fn assignments(&self) -> &[Assignment<N>] { &self.assignments }
  pub fn value(&self) -> &ExpressionMeta<N> { &self.value }

  pub fn add_assignment(&mut self, asgn: Assignment<N>) { self.assignments.push(asgn); }
  pub fn set_value(&mut self, val: ExpressionMeta<N>) { self.value = Box::new(val); }
}

#[derive(Clone, Debug)]
pub struct Assignment<N: Enhancement> {
  pos: Position,
  lhs: Destructure,
  rhs: ExpressionMeta<N>
}

impl<N: Enhancement> Assignment<N> {
  pub fn new(pos: Position, lhs: Destructure, rhs: ExpressionMeta<N>) -> Assignment<N> { Assignment { pos, lhs, rhs } }
  pub fn lhs(&self) -> &Destructure { &self.lhs }
  pub fn rhs(&self) -> &ExpressionMeta<N> { &self.rhs }
  pub fn pos(&self) -> &Position { &self.pos }
}

/// A destructuring target, e.g. `{a: v1, b: v2} = {a: 1, b: "hi"}`
#[derive(Clone, Debug)]
pub enum Destructure {
  Blank,
  Identifier(String),
  Array(Vec<Destructure>),
  Object(Vec<(String, Destructure)>)
}

impl Destructure {
  pub fn is_blank(&self) -> bool { matches!(self, Self::Blank) }
  pub fn as_identifier(&self) -> Option<&str> { pick_opt!(self, Self::Identifier(v) => v) }
  pub fn as_array(&self) -> Option<&[Destructure]> { pick_opt!(self, Self::Array(v) => v) }
  pub fn as_object(&self) -> Option<&[(String, Destructure)]> { pick_opt!(self, Self::Object(v) => v) }

  pub fn identifier_count(&self) -> usize {
    match self {
      Self::Blank => 0,
      Self::Identifier(_) => 1,
      Self::Array(a) => a.iter().map(|d| d.identifier_count()).sum(),
      Self::Object(a) => a.iter().map(|p| p.1.identifier_count()).sum()
    }
  }

  pub fn identifiers(&self) -> impl Iterator<Item = &String> + '_ {
    match self {
      Self::Blank => Vec::new().into_iter(),
      Self::Identifier(s) => vec![s].into_iter(),
      Self::Array(a) => a.iter().flat_map(|d| d.identifiers()).collect::<Vec<_>>().into_iter(),
      Self::Object(a) => a.iter().flat_map(|p| p.1.identifiers()).collect::<Vec<_>>().into_iter()
    }
  }
}

#[derive(Clone, Debug)]
pub struct ExpressionMeta<N: Enhancement> {
  expr: Expression<N>,
  pos: Position
}

impl<N: Enhancement> ExpressionMeta<N> {
  pub fn new(expr: Expression<N>, pos: Position) -> ExpressionMeta<N> { ExpressionMeta { expr, pos } }
  pub fn expr(&self) -> &Expression<N> { &self.expr }
  pub fn pos(&self) -> &Position { &self.pos }
  pub fn into_expr(self) -> Expression<N> { self.expr }
  pub fn allowed_skip_semi(&self) -> bool { self.expr.allowed_skip_semi() }

  pub fn into_destructure(self) -> (Destructure, Position) { (self.expr.into_destructure(), self.pos) }
}

/// An enhanced expression.
#[derive(Clone, Debug)]
pub enum Expression<N: Enhancement> {
  Variable(String),
  Literal(Literal),
  Unary(UnaryOp, Box<ExpressionMeta<N>>),
  Binary(BinaryOp, Box<ExpressionMeta<N>>, Box<ExpressionMeta<N>>),
  Array(Vec<ExpressionMeta<N>>),
  Object(Vec<(String, ExpressionMeta<N>)>),
  IfBlock(Box<ExpressionMeta<N>>, Block<N>, Vec<(ExpressionMeta<N>, Block<N>)>, Block<N>),
  Function(Vec<Destructure>, Block<N>, N::Function),
  Dot(Box<ExpressionMeta<N>>, DotIndex),
  Index(Box<ExpressionMeta<N>>, u64),
  And(Box<ExpressionMeta<N>>, Box<ExpressionMeta<N>>),
  Or(Box<ExpressionMeta<N>>, Box<ExpressionMeta<N>>),
  Call(Box<ExpressionMeta<N>>, Vec<ExpressionMeta<N>>)
}

impl<N: Enhancement> Expression<N> {
  pub fn as_variable(&self) -> Option<&str> { pick_opt!(self, Self::Variable(s) => s) }
  pub fn as_literal(&self) -> Option<&Literal> { pick_opt!(self, Self::Literal(v) => v) }
  pub fn allowed_skip_semi(&self) -> bool { matches!(self, Self::IfBlock(..)) }

  /// Convert once the parser has realized that the expression we're parsing is actually the lhs of an
  /// assignement.
  pub fn into_destructure(self) -> Destructure {
    match self {
      Self::Variable(s) => {
        if s == "_" {
          Destructure::Blank
        } else {
          Destructure::Identifier(s)
        }
      }
      Self::Array(a) => Destructure::Array(a.into_iter().map(|e| e.into_destructure().0).collect()),
      Self::Object(o) => Destructure::Object(o.into_iter().map(|(k, e)| (k, e.into_destructure().0)).collect()),
      e => panic!("Can't convert into target {:?}", e)
    }
  }
}

#[derive(Copy, Clone, Debug)]
pub enum UnaryOp {
  Negate,
  Not
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryOp {
  Add,
  Subtract,
  Multiply,
  Divide,
  Mod,
  Gt,
  Gte,
  Lt,
  Lte,
  Equals,
  NotEquals
}

#[derive(Clone, Debug)]
pub enum DotIndex {
  Identifier(String),
  Integer(u64)
}

pub trait Enhancement: fmt::Debug {
  type Function: fmt::Debug + Clone;
}

impl Enhancement for () {
  type Function = ();
}

pub trait UseCount {
  fn is_single_use(&self) -> bool;
}

impl UseCount for () {
  fn is_single_use(&self) -> bool { false }
}

pub struct Scope<T: UseCount> {
  locals: Locals<T>
}

impl<T: Clone + Default + fmt::Debug + UseCount> Default for Scope<T> {
  fn default() -> Scope<T> { Scope::new() }
}

impl<T: Clone + Default + fmt::Debug + UseCount> Scope<T> {
  pub fn new() -> Scope<T> { Scope { locals: Locals::new() } }
  pub fn restore_used(&mut self) { self.locals.restore_used(); }
  pub fn reserve_used(&mut self) { self.locals.reserve_used(); }
  pub fn resolve_local(&mut self, name: &str) -> Option<(usize, T)> { self.locals.resolve_local(name) }
  pub fn add_local(&mut self, name: String) { self.locals.add_local(Local::new(name)); }
  pub fn mark_initialized(&mut self, neg_ind: usize, t: T) { self.locals.mark_initialized(neg_ind, t); }
  pub fn begin_scope(&mut self) { self.locals.incr_depth(); }
  pub fn end_scope(&mut self) { self.locals.decr_depth_and_truncate(); }
}

pub struct Locals<T> {
  locals: Vec<Local<T>>,
  scope_depth: u16
}

impl<T: Clone + Default + fmt::Debug + UseCount> Default for Locals<T> {
  fn default() -> Locals<T> { Locals::new() }
}

impl<T: Clone + Default + fmt::Debug + UseCount> Locals<T> {
  pub fn new() -> Locals<T> { Locals { locals: Vec::new(), scope_depth: 0 } }

  pub fn depth(&self) -> u16 { self.scope_depth }
  pub fn incr_depth(&mut self) { self.scope_depth += 1; }

  pub fn decr_depth_and_truncate(&mut self) {
    self.scope_depth -= 1;
    if let Some(p) = self.locals.iter().position(|l| l.depth() > self.scope_depth) {
      self.locals.truncate(p);
    }
  }

  pub fn drain(&mut self) -> Vec<Local<T>> {
    if let Some(p) = self.locals.iter().position(|l| l.depth() > self.scope_depth) {
      self.locals.split_off(p)
    } else {
      Vec::new()
    }
  }

  pub fn add_local(&mut self, local: Local<T>) {
    if self.locals.len() >= MAX_LOCALS {
      panic!("Too many locals: {}", self.locals.len());
    }

    self.locals.push(local);
  }

  pub fn resolve_local(&mut self, name: &str) -> Option<(usize, T)> {
    // Prefer initialized locals.
    let mut l = self.locals.iter_mut().enumerate().rev().find(|(_, l)| l.name() == name && l.is_initialized());

    // Fall back to uninititalized to get a good error message.
    if l.is_none() {
      l = self.locals.iter_mut().enumerate().rev().find(|(_, l)| l.name() == name);
    }

    match l {
      None => None,
      Some((i, local)) => {
        if local.is_initialized() {
          let used = local.incr_used(self.scope_depth);

          let ltype = local.local_type();
          if ltype.is_single_use() && used > 1 {
            panic!("Can't use variable \"{}\" (type {:?}) more than once.", name, ltype);
          }

          // The bottom of stack is reserved for the current function, so add one to the index.
          Some((i + 1, local.local_type().clone()))
        } else {
          panic!("Can't read local \"{}\" in its own initializer.", name)
        }
      }
    }
  }

  pub fn mark_initialized(&mut self, neg_ind: usize, t: T) {
    let locals_len = self.locals.len();
    let local = &mut self.locals[locals_len - neg_ind];
    local.mark_initialized(self.scope_depth, t);
  }

  pub fn set_captured(&mut self, locals_ind: usize, captured: bool) { self.locals[locals_ind].set_captured(captured); }

  pub fn reserve_used(&mut self) {
    for local in &mut self.locals {
      local.reserve_used(self.scope_depth);
    }
  }

  pub fn restore_used(&mut self) {
    for local in &mut self.locals {
      local.restore_used(self.scope_depth);
    }
  }
}

#[derive(Debug)]
pub struct Local<T> {
  name: String,               // the text of the local
  depth: u16,                 // the scope depth of declaration
  is_captured: bool,          // if the local is closed
  used: Vec<u16>,             // use count: index is depth, relative to declaration
  reserved: Option<Vec<u16>>, // temp use count for if/else
  local_type: T
}

impl<T: Clone + Default + fmt::Debug + UseCount> Local<T> {
  pub fn new(name: String) -> Local<T> {
    Local { name, depth: 0, is_captured: false, used: Vec::new(), reserved: None, local_type: Default::default() }
  }

  pub fn name(&self) -> &str { &self.name }
  pub fn depth(&self) -> u16 { self.depth }
  pub fn is_captured(&self) -> bool { self.is_captured }
  pub fn set_captured(&mut self, cap: bool) { self.is_captured = cap }
  pub fn local_type(&self) -> &T { &self.local_type }

  /// Increment the usage counter at a particular depth.
  pub fn incr_used(&mut self, depth: u16) -> u16 {
    let above = (depth - self.depth) as usize;
    if self.used.len() <= above {
      self.used.resize(above + 1, 0);
    }
    self.used[above] += 1;
    self.used.iter().sum()
  }

  /// Move usage counts of higher-depth into a temporary holding; useful for if/else blocks or other tenative
  /// use counting.
  pub fn reserve_used(&mut self, depth: u16) {
    let above = (depth - self.depth) as usize;
    if self.used.len() > above {
      if let Some(reserved) = &mut self.reserved {
        reserved.resize(max(reserved.len(), self.used.len()), 0);
        for (i, u) in self.used.drain(above ..).enumerate() {
          reserved[above + i] = max(reserved[above + i], u);
        }
      } else {
        self.reserved = Some(vec![0; above].into_iter().chain(self.used.drain(above ..)).collect());
      }
    }
  }

  /// Restore usage counts from temporary holding. Opposite of `reserve_used`.
  pub fn restore_used(&mut self, depth: u16) {
    let above = (depth - self.depth) as usize;
    if let Some(reserved) = &mut self.reserved {
      assert!(self.used.len() == above);
      self.used.extend(reserved.drain(above ..));
    }
  }

  pub fn mark_initialized(&mut self, scope_depth: u16, t: T) {
    self.depth = scope_depth;
    self.local_type = t;
  }

  pub fn is_initialized(&self) -> bool { self.depth > 0 }
}

pub struct Rule {
  prefix: Prefix,
  infix: Infix,
  precedence: Precedence
}

impl Rule {
  pub fn new(prefix: Prefix, infix: Infix, precedence: Precedence) -> Rule { Rule { prefix, infix, precedence } }

  pub fn prefix(&self) -> Prefix { self.prefix }
  pub fn infix(&self) -> Infix { self.infix }
  pub fn precedence(&self) -> Precedence { self.precedence }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Precedence {
  None,
  Or,
  And,
  Equality,
  Comparison,
  Term,
  Factor,
  Unary,
  Call,
  Primary
}

impl Precedence {
  pub fn up(&self) -> Precedence {
    match self {
      Self::None => Self::Or,
      Self::Or => Self::And,
      Self::And => Self::Equality,
      Self::Equality => Self::Comparison,
      Self::Comparison => Self::Term,
      Self::Term => Self::Factor,
      Self::Factor => Self::Unary,
      Self::Unary => Self::Call,
      Self::Call => Self::Primary,
      Self::Primary => panic!("No precedence higher than primary.")
    }
  }
}

// Wrapper methods, so that they can be indexed by rule.

fn variable(compiler: &mut Parser) -> TFut<'_> { Box::pin(compiler.variable()) }
fn unary(compiler: &mut Parser) -> TFut<'_> { Box::pin(compiler.unary()) }
fn literal(compiler: &mut Parser) -> TFut<'_> { Box::pin(compiler.literal()) }
fn array(compiler: &mut Parser) -> TFut<'_> { Box::pin(compiler.array()) }
fn object(compiler: &mut Parser) -> TFut<'_> { Box::pin(compiler.object()) }
fn grouping(compiler: &mut Parser) -> TFut<'_> { Box::pin(compiler.grouping()) }
fn if_block(compiler: &mut Parser) -> TFut<'_> { Box::pin(compiler.if_block()) }
fn function(compiler: &mut Parser) -> TFut<'_> { Box::pin(compiler.function()) }
fn binary(compiler: &mut Parser, expr: ExpressionMeta<()>) -> TFut<'_> { Box::pin(compiler.binary(expr)) }
fn index(compiler: &mut Parser, expr: ExpressionMeta<()>) -> TFut<'_> { Box::pin(compiler.index(expr)) }
fn dot(compiler: &mut Parser, expr: ExpressionMeta<()>) -> TFut<'_> { Box::pin(compiler.dot(expr)) }
fn and(compiler: &mut Parser, expr: ExpressionMeta<()>) -> TFut<'_> { Box::pin(compiler.and(expr)) }
fn or(compiler: &mut Parser, expr: ExpressionMeta<()>) -> TFut<'_> { Box::pin(compiler.or(expr)) }
fn call(compiler: &mut Parser, expr: ExpressionMeta<()>) -> TFut<'_> { Box::pin(compiler.call(expr)) }
fn call_unit(compiler: &mut Parser, expr: ExpressionMeta<()>) -> TFut<'_> { Box::pin(compiler.call_unit(expr)) }

fn construct_rules() -> HashMap<TokenType, Rule> {
  let mut rules = HashMap::new();

  rules.insert(TokenType::Bof, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::Eof, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::Comma, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::Equals, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::Semi, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::Colon, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::OpenCurl, Rule::new(Some(object), None, Precedence::None));
  rules.insert(TokenType::CloseCurl, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::OpenSquare, Rule::new(Some(array), Some(index), Precedence::Call));
  rules.insert(TokenType::CloseSquare, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::DoubleAnd, Rule::new(None, Some(and), Precedence::And));
  rules.insert(TokenType::DoubleOr, Rule::new(None, Some(or), Precedence::Or));
  rules.insert(TokenType::Gt, Rule::new(None, Some(binary), Precedence::Comparison));
  rules.insert(TokenType::Lt, Rule::new(None, Some(binary), Precedence::Comparison));
  rules.insert(TokenType::Gte, Rule::new(None, Some(binary), Precedence::Comparison));
  rules.insert(TokenType::Lte, Rule::new(None, Some(binary), Precedence::Comparison));
  rules.insert(TokenType::DoubleEq, Rule::new(None, Some(binary), Precedence::Equality));
  rules.insert(TokenType::NotEq, Rule::new(None, Some(binary), Precedence::Equality));
  rules.insert(TokenType::Plus, Rule::new(None, Some(binary), Precedence::Term));
  rules.insert(TokenType::Minus, Rule::new(Some(unary), Some(binary), Precedence::Term));
  rules.insert(TokenType::Star, Rule::new(None, Some(binary), Precedence::Factor));
  rules.insert(TokenType::Slash, Rule::new(None, Some(binary), Precedence::Factor));
  rules.insert(TokenType::Percent, Rule::new(None, Some(binary), Precedence::Factor));
  rules.insert(TokenType::Bang, Rule::new(Some(unary), None, Precedence::Unary));
  rules.insert(TokenType::OpenParen, Rule::new(Some(grouping), Some(call), Precedence::Call));
  rules.insert(TokenType::CloseParen, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::Dot, Rule::new(None, Some(dot), Precedence::Call));
  rules.insert(TokenType::FnWord, Rule::new(Some(function), None, Precedence::None));
  rules.insert(TokenType::IfWord, Rule::new(Some(if_block), None, Precedence::None));
  rules.insert(TokenType::ElseifWord, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::ElseWord, Rule::new(None, None, Precedence::None));
  rules.insert(TokenType::TrueLit, Rule::new(Some(literal), None, Precedence::None));
  rules.insert(TokenType::FalseLit, Rule::new(Some(literal), None, Precedence::None));
  rules.insert(TokenType::IntLit, Rule::new(Some(literal), None, Precedence::None));
  rules.insert(TokenType::FloatLit, Rule::new(Some(literal), None, Precedence::None));
  rules.insert(TokenType::StringLit, Rule::new(Some(literal), None, Precedence::None));
  rules.insert(TokenType::Unit, Rule::new(Some(literal), Some(call_unit), Precedence::Call));
  rules.insert(TokenType::Identifier, Rule::new(Some(variable), None, Precedence::None));
  rules.insert(TokenType::Error, Rule::new(None, None, Precedence::None));

  rules
}

fn to_value<V>(v: &str) -> Literal
where
  V: Into<Literal> + FromStr,
  Error: From<<V as FromStr>::Err>
{
  v.parse::<V>().map_err(Error::from).unwrap().into()
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::test::setup;

  #[tokio::test]
  async fn literal() {
    setup();
    let data = r#"2"#;
    let ast = parse(data).await;
    assert!(ast.block().assignments().is_empty());
    assert_eq!(ast.block().value().expr().as_literal().unwrap().as_int(), 2);
  }

  #[tokio::test]
  async fn assignment() {
    setup();
    let data = r#"a=1; 0"#;
    let ast = parse(data).await;
    let assgn = &ast.block().assignments()[0];
    assert_eq!(assgn.lhs().as_identifier().unwrap(), "a");
    assert_eq!(assgn.rhs().expr().as_literal().unwrap().as_int(), 1);
  }

  #[tokio::test]
  async fn fn_parses() {
    setup();
    let data = r#"fn(a) { 0 }"#;
    let _ = parse(data).await;
  }

  #[tokio::test]
  async fn fn_destruct_params_parses() {
    setup();
    let data = r#"fn({a, b, c}) {{a, b, c}}"#;
    let _ = parse(data).await;
  }

  #[tokio::test]
  async fn single_use() {
    setup();
    let data = r#"a = "SINGLE"; b = a; c = a;"#;
    let _ = parse(data).await;
  }
}
