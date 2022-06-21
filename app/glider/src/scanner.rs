//! The scanner for the Glider language.

use crate::pick;
use std::fmt;
use std::str::CharIndices;
use std::vec::IntoIter;

pub struct Scanner<'s> {
  was_dot: bool,
  input: &'s str,
  iter: CharIndices<'s>,
  pos: Position
}

impl<'s> Scanner<'s> {
  pub fn new(input: &'s str) -> Scanner {
    Scanner { input, iter: input.char_indices(), pos: Position::zero(), was_dot: false }
  }
}

impl<'s> Iterator for Scanner<'s> {
  type Item = TokenMeta;

  fn next(&mut self) -> Option<TokenMeta> {
    self.skip_whitespace();
    let pos = self.pos.clone();
    self.incr().map(|(st, c)| {
      let token_type = match c {
        '%' => Token::Percent,
        '(' => self.if_next(')', Token::Unit, Token::OpenParen),
        ')' => Token::CloseParen,
        '*' => Token::Star,
        '+' => Token::Plus,
        '-' => Token::Minus,
        '.' => Token::Dot,
        '/' => Token::Slash,
        ':' => Token::Colon,
        ';' => Token::Semi,
        '[' => Token::OpenSquare,
        ']' => Token::CloseSquare,
        '{' => Token::OpenCurl,
        '}' => Token::CloseCurl,
        ',' => Token::Comma,
        '!' => self.if_next('=', Token::NotEq, Token::Bang),
        '<' => self.if_next('=', Token::Lte, Token::Lt),
        '=' => self.if_next('=', Token::DoubleEq, Token::Equals),
        '>' => self.if_next('=', Token::Gte, Token::Gt),
        '"' => self.next_string(st),
        '&' => self.if_next('&', Token::DoubleAnd, Token::Error(format!("Incomplete && at {}.", &self.pos))),
        '|' => self.if_next('|', Token::DoubleOr, Token::Error(format!("Incomplete || at {}.", &self.pos))),
        c if c.is_digit(10) => self.next_number(st),
        c if c.is_ascii_alphabetic() || c == '_' => self.next_identifier(st),
        _ => Token::Error(format!("Unexpected character at {}.", &self.pos))
      };
      self.was_dot = token_type == Token::Dot;

      TokenMeta::new(token_type, pos.clone())
    })
  }
}

impl<'s> Scanner<'s> {
  pub fn drain_into_iter(&mut self) -> IntoIter<TokenMeta> { self.collect::<Vec<_>>().into_iter() }

  pub fn pos(&self) -> &Position { &self.pos }

  fn incr(&mut self) -> Option<(usize, char)> {
    self.pos.incr();
    self.iter.next()
  }

  fn next_identifier(&mut self, open: usize) -> Token {
    let mut close = open + 1;
    while let Some((i, c)) = self.peek() {
      close = i;
      if c.is_ascii_alphabetic() || c.is_digit(10) || c == '_' {
        close += 1;
        self.incr();
      } else {
        break;
      }
    }

    match &self.input[open .. close] {
      "fn" => Token::FnWord,
      "if" => Token::IfWord,
      "elseif" => Token::ElseifWord,
      "else" => Token::ElseWord,
      "false" => Token::FalseLit,
      "true" => Token::TrueLit,
      id => Token::Identifier(id.to_string())
    }
  }

  fn next_number(&mut self, open: usize) -> Token {
    let mut is_int = true;
    let mut has_trail = false;
    let mut close = open + 1;
    while let Some((i, c)) = self.peek() {
      close = i;
      if !c.is_digit(10) && c != '.' {
        break;
      } else if c == '.' {
        if self.was_dot {
          break;
        }
        if !is_int {
          return Token::Error(format!("Illegal number at {}.", open));
        }
        is_int = false;
        has_trail = false;
        close += 1;
        self.incr();
      } else {
        has_trail = true;
        close += 1;
        self.incr();
      }
    }

    if is_int {
      Token::IntLit(self.input[open .. close].to_string())
    } else if has_trail {
      Token::FloatLit(self.input[open .. close].to_string())
    } else {
      Token::Error(format!("Bad number starting at {}.", open))
    }
  }

  fn next_string(&mut self, open: usize) -> Token {
    while let Some((i, c)) = self.incr() {
      if c == '"' {
        return Token::StringLit(self.input[open + 1 .. i].to_string());
      } else if c == '\n' {
        self.pos.newline();
      }
    }
    Token::Error(format!("Unterminated string starting at {}.", open))
  }

  fn peek(&self) -> Option<(usize, char)> { self.iter.clone().next() }

  /// Skip whitespace and comments.
  fn skip_whitespace(&mut self) {
    while let Some((_, c)) = self.peek() {
      if c.is_whitespace() {
        self.incr();
        if c == '\n' {
          self.pos.newline();
        }
      } else if c == '/' {
        if self.iter.as_str().starts_with("//") {
          self.incr();
          while !matches!(self.peek(), Some((_, '\n')) | None) {
            self.incr();
          }
        } else {
          break;
        }
      } else {
        break;
      }
    }
  }

  fn if_next(&mut self, t: char, if_t: Token, if_f: Token) -> Token {
    match self.peek() {
      Some((_, x)) if x == t => {
        self.incr();
        if_t
      }
      _ => if_f
    }
  }
}

/// The token, as well as identifying information about where the token came from.
#[derive(Debug, Clone)]
pub struct TokenMeta {
  pos: Position,
  token: Token
}

impl TokenMeta {
  pub fn new(token: Token, pos: Position) -> TokenMeta { TokenMeta { token, pos } }

  pub fn is_error(&self) -> bool { self.token.is_error() }
  pub fn pos(&self) -> &Position { &self.pos }
  pub fn token(&self) -> &Token { &self.token }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
  Bof,
  Eof,
  Comma,
  Equals,
  Semi,
  Colon,
  OpenCurl,
  CloseCurl,
  OpenSquare,
  CloseSquare,
  DoubleAnd,
  DoubleOr,
  Gt,
  Lt,
  Gte,
  Lte,
  DoubleEq,
  NotEq,
  Plus,
  Minus,
  Star,
  Slash,
  Percent,
  Bang,
  OpenParen,
  CloseParen,
  Dot,
  FnWord,
  IfWord,
  ElseifWord,
  ElseWord,
  TrueLit,
  FalseLit,
  Unit,
  IntLit(String),
  FloatLit(String),
  StringLit(String),
  Identifier(String),
  Error(String)
}

impl Token {
  pub fn as_identifier(&self) -> &str { pick!(self, Self::Identifier(s) => s, "Not an identifier: {:?}") }

  pub fn is_error(&self) -> bool { matches!(self, Token::Error(_)) }

  pub fn tt(&self) -> TokenType {
    match self {
      Self::Bof => TokenType::Bof,
      Self::Eof => TokenType::Eof,
      Self::Comma => TokenType::Comma,
      Self::Equals => TokenType::Equals,
      Self::Semi => TokenType::Semi,
      Self::Colon => TokenType::Colon,
      Self::OpenCurl => TokenType::OpenCurl,
      Self::CloseCurl => TokenType::CloseCurl,
      Self::OpenSquare => TokenType::OpenSquare,
      Self::CloseSquare => TokenType::CloseSquare,
      Self::DoubleAnd => TokenType::DoubleAnd,
      Self::DoubleOr => TokenType::DoubleOr,
      Self::Gt => TokenType::Gt,
      Self::Lt => TokenType::Lt,
      Self::Gte => TokenType::Gte,
      Self::Lte => TokenType::Lte,
      Self::DoubleEq => TokenType::DoubleEq,
      Self::NotEq => TokenType::NotEq,
      Self::Plus => TokenType::Plus,
      Self::Minus => TokenType::Minus,
      Self::Star => TokenType::Star,
      Self::Slash => TokenType::Slash,
      Self::Percent => TokenType::Percent,
      Self::Bang => TokenType::Bang,
      Self::OpenParen => TokenType::OpenParen,
      Self::CloseParen => TokenType::CloseParen,
      Self::Dot => TokenType::Dot,
      Self::FnWord => TokenType::FnWord,
      Self::IfWord => TokenType::IfWord,
      Self::ElseifWord => TokenType::ElseifWord,
      Self::ElseWord => TokenType::ElseWord,
      Self::TrueLit => TokenType::TrueLit,
      Self::FalseLit => TokenType::FalseLit,
      Self::Unit => TokenType::Unit,
      Self::IntLit(..) => TokenType::IntLit,
      Self::FloatLit(..) => TokenType::FloatLit,
      Self::StringLit(..) => TokenType::StringLit,
      Self::Identifier(..) => TokenType::Identifier,
      Self::Error(..) => TokenType::Error
    }
  }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TokenType {
  Bof,
  Eof,
  Comma,
  Equals,
  Semi,
  Colon,
  OpenCurl,
  CloseCurl,
  OpenSquare,
  CloseSquare,
  DoubleAnd,
  DoubleOr,
  Gt,
  Lt,
  Gte,
  Lte,
  DoubleEq,
  NotEq,
  Plus,
  Minus,
  Star,
  Slash,
  Percent,
  Bang,
  OpenParen,
  CloseParen,
  Dot,
  FnWord,
  IfWord,
  ElseifWord,
  ElseWord,
  TrueLit,
  FalseLit,
  IntLit,
  FloatLit,
  StringLit,
  Unit,
  Identifier,
  Error
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Position {
  line: u16,
  col: u16,
  chr: u32
}

impl fmt::Display for Position {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}:{}", self.line + 1, self.col + 1) }
}

impl Position {
  pub fn zero() -> Position { Position { line: 0, col: 0, chr: 0 } }
  pub fn at(line: u16, col: u16, chr: u32) -> Position { Position { line, col, chr } }

  pub fn line(&self) -> u16 { self.line }
  pub fn col(&self) -> u16 { self.col }
  pub fn chr(&self) -> u32 { self.chr }

  pub fn newline(&mut self) {
    self.line += 1;
    self.col = 0;
  }

  pub fn incr(&mut self) {
    self.chr += 1;
    self.col += 1;
  }
}
