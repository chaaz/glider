//! The actual VM for executing some bytecode.

use crate::callable::{FnDefInner, NativeDefInner, RunData};
use crate::compiler::{Chunk, FnInd, Opcode};
use crate::custom::Custom;
use crate::scanner::Position;
use crate::value::Value;
use std::fmt::write;
use std::iter::once;
use std::sync::{Arc, Mutex};
use tracing::{debug, enabled, Level};

type Stack<C> = Vec<Value<C>>;
type Frames = Vec<Frame>;

pub async fn run<C: Custom>(f: Value<C>, fn_ind: FnInd, args: Vec<Value<C>>, status: C::Status) -> Value<C> {
  match f {
    Value::FnDef(f, cv) => run_fn_def(f, cv, fn_ind, args, status).await,
    Value::NativeDef(f, cv, cc) => run_ntv_def(f, cv, cc, fn_ind, args, status).await,
    other => panic!("Not a callable value: {:?}", other)
  }
}

pub async fn run_script<C: Custom>(script: Arc<Mutex<FnDefInner<C>>>, cv: Vec<Value<C>>) -> Value<C> {
  run_fn_def(script, cv, FnInd::script(), Vec::new(), Default::default()).await
}

pub async fn run_fn_def<C: Custom>(
  fn_def: Arc<Mutex<FnDefInner<C>>>, cv: Vec<Value<C>>, fn_ind: FnInd, args: Vec<Value<C>>, status: C::Status
) -> Value<C> {
  assert_eq!(fn_ind.extracts().len(), args.len());
  let mut args: Vec<_> = fn_ind.extracts().iter().zip(args.into_iter()).flat_map(|(e, a)| e.extracted(a)).collect();
  run_with_at(fn_def, fn_ind.index(), &mut args, cv, status).await
}

async fn run_with_at<C: Custom>(
  fn_def: Arc<Mutex<FnDefInner<C>>>, fn_ind: usize, args: &mut [Value<C>], capts: Vec<Value<C>>, status: C::Status
) -> Value<C> {
  let mut vm = Vm::new_with_args(fn_def, args, capts, status);
  vm.go(FrameOp::Push(Frame::new(0, fn_ind, false, Position::zero()))).await
}

pub async fn run_ntv_def<C: Custom>(
  ntv_def: Arc<Mutex<NativeDefInner<C>>>, cv: Vec<Value<C>>, cc: C::Capture, fn_ind: FnInd, args: Vec<Value<C>>,
  status: C::Status
) -> Value<C> {
  let (d_impl, run_fn) = {
    let inner = ntv_def.try_lock().unwrap();
    (inner.completed_at(fn_ind.index()), *inner.run_fn())
  };
  let build_data = d_impl.borrow().data().clone();
  let run_data = RunData::new(Position::zero(), status, cc);
  (run_fn)(args, cv, build_data, run_data).await
}

pub struct Vm<C: Custom> {
  status: C::Status,
  stack: Stack<C>,
  frames: Frames
}

impl<C: Custom> Vm<C> {
  pub fn new_with_args(
    script: Arc<Mutex<FnDefInner<C>>>, args: &mut [Value<C>], capts: Vec<Value<C>>, status: C::Status
  ) -> Vm<C> {
    let script_val = Value::FnDef(script, capts);

    let mut vm =
      Vm { status, stack: once(script_val).chain(args.iter_mut().map(|a| a.shift())).collect(), frames: Vec::new() };
    vm.push_captures(0);
    vm
  }

  pub fn new(script: Arc<Mutex<FnDefInner<C>>>) -> Vm<C> {
    Vm::new_with_args(script, &mut [], Vec::new(), Default::default())
  }

  async fn go(&mut self, op: FrameOp) -> Value<C> {
    let mut operated_go = op.operate(&mut self.frames);

    while self.should_go(operated_go) {
      let frame = self.frames.last().unwrap();
      let call_pos = frame.pos().clone();

      // When glider makes a call, we retreat back to this loop with a frame operation to unlock the chunk, so
      // that we can make the next in-code or native call.

      // TODO(later): cleanup fn / native data at end of compile.
      operated_go = match self.stack.get_mut(frame.offset()).unwrap() {
        Value::FnDef(fn_def, _capts) => {
          let fn_impl = fn_def.try_lock().unwrap().completed_at(frame.morph_index());
          // We need to read a cloned chunk so that the `Future` this function returns is `Send`
          let chunk = fn_impl.borrow().chunk().clone();
          self.run(&chunk).await.operate(&mut self.frames)
        }
        Value::NativeDef(natv_def, capts, ec) => {
          let offset = frame.offset();
          // We need to read cloned data so that the `Future` this function returns is `Send`
          let (natv_impl, capts, ec, run_fn) = {
            let natv_def = natv_def.clone();
            let natv_def = natv_def.try_lock().unwrap();
            let natv_impl = natv_def.completed_at(frame.morph_index());
            let run_fn = *natv_def.run_fn();
            (natv_impl, capts.iter_mut().map(|v| v.shift()).collect(), ec.clone(), run_fn)
          };
          let stack_len = self.stack.len();
          let args = self.stack[offset + 1 .. stack_len].iter_mut().map(|v| v.shift()).collect();
          let build_data = natv_impl.borrow().data().clone();
          let run_data = RunData::new(call_pos, self.status.clone(), ec);
          let result = (run_fn)(args, capts, build_data, run_data).await;
          self.stack[offset] = result;
          self.stack.truncate(offset + 1);
          self.frames.pop().unwrap();
          true
        }
        v => panic!("Can't execute as function: {:?}", v)
      };
    }

    self.stack.pop().unwrap()
  }

  pub fn should_go(&self, operated_go: bool) -> bool { operated_go && !self.frames.is_empty() }

  async fn run(&mut self, chunk: &Chunk<C>) -> FrameOp {
    while let Some(opcode) = chunk.at(self.ip()) {
      self.debug_handle(opcode);

      match opcode {
        Opcode::Return => {
          let offset = self.frame().offset();
          self.stack.swap_remove(offset);
          self.stack.truncate(offset + 1);
          return FrameOp::Pop;
        }
        Opcode::Negate => self.stack_replace(|v| v.op_negate()),
        Opcode::Not => self.stack_replace(|v| v.op_not()),
        Opcode::Constant(c) => self.constant(c, chunk),
        Opcode::Closure(c, capture_inds) => self.closure(c, capture_inds, chunk),
        Opcode::Add => binary(&mut self.stack, |v, w| v.op_add(w)),
        Opcode::Subtract => binary(&mut self.stack, |v, w| v.op_subtract(w)),
        Opcode::Multiply => binary(&mut self.stack, |v, w| v.op_multiply(w)),
        Opcode::Divide => binary(&mut self.stack, |v, w| v.op_divide(w)),
        Opcode::Mod => binary(&mut self.stack, |v, w| v.op_mod(w)),
        Opcode::And => binary(&mut self.stack, |v, w| v.op_and(w)),
        Opcode::Or => binary(&mut self.stack, |v, w| v.op_or(w)),
        Opcode::Lt => binary(&mut self.stack, |v, w| v.op_lt(w)),
        Opcode::Lte => binary(&mut self.stack, |v, w| v.op_lte(w)),
        Opcode::Gt => binary(&mut self.stack, |v, w| v.op_gt(w)),
        Opcode::Gte => binary(&mut self.stack, |v, w| v.op_gte(w)),
        Opcode::Equals => binary(&mut self.stack, |v, w| v.op_eq(w)),
        Opcode::NotEquals => binary(&mut self.stack, |v, w| v.op_neq(w)),
        Opcode::GetLocal(l) => {
          let offset = self.frame().offset();
          let v = self.stack.get_mut(*l + offset).unwrap().shift();
          self.stack.push(v);
        }
        Opcode::GetIndex(i) => {
          let l = self.stack.len() - 1;
          self.stack[l] = self.stack[l].as_array_mut()[*i].shift();
        }
        Opcode::GetJsonIndex(i) => {
          let l = self.stack.len() - 1;
          self.stack[l] = Value::Json(self.stack[l].as_json_mut().as_array_mut().unwrap().swap_remove(*i))
        }
        Opcode::GetJsonKey(s) => {
          let l = self.stack.len() - 1;
          self.stack[l] = Value::Json(self.stack[l].as_json_mut().as_object_mut().unwrap().remove(s).unwrap())
        }
        Opcode::Extract(extract) => {
          let arr = self.stack.pop().unwrap();
          for target in extract.extracted(arr) {
            self.stack.push(target);
          }
        }
        Opcode::Array(len) => {
          let stack_len = self.stack.len();
          let mut parts = Vec::new();
          for i in 0 .. *len {
            parts.push(self.stack[stack_len - (len - i)].shift());
          }
          if *len > 0 {
            self.stack[stack_len - len] = Value::Array(parts);
            self.stack.truncate(stack_len - (len - 1));
          } else {
            self.stack.push(Value::Array(parts));
          }
        }
        Opcode::Object(inds) => {
          let stack_len = self.stack.len();
          let inds_len = inds.len();
          let mut parts = Vec::new();
          parts.resize_with(inds_len, || Value::Int(0));
          for (i, ind) in inds.iter().enumerate() {
            parts[*ind] = self.stack[stack_len - (inds_len - i)].shift();
          }
          if inds_len > 0 {
            self.stack[stack_len - inds_len] = Value::Array(parts);
            self.stack.truncate(stack_len - (inds_len - 1));
          } else {
            self.stack.push(Value::Array(parts));
          }
        }
        Opcode::Jump(offset) => self.incr_ip(*offset),
        Opcode::JumpIfFalse(offset) => {
          if !self.stack.last().unwrap().as_bool() {
            self.incr_ip(*offset)
          }
        }
        Opcode::Pop => drop(self.stack.pop()),
        // TODO(performance): pulling from positions might make us swap outside the opcodes/constants :(
        Opcode::Call(morph_index, argc) => return self.call(*morph_index, *argc, chunk.pos_at(self.ip()).unwrap())
      }

      self.incr_ip(1);
    }

    // We shouldn't get here
    panic!("Ran out of code");
  }

  fn constant(&mut self, c: &usize, chunk: &Chunk<C>) {
    let c = *c;
    let c_val = chunk.constant_at(c).unwrap_or_else(|| panic!("No constant at {}", c));
    let c_val = c_val.to_value();
    self.stack_push(c_val);
  }

  fn closure(&mut self, c: &usize, capture_inds: &[usize], chunk: &Chunk<C>) {
    self.constant(c, chunk);

    let offset = self.frames.last().unwrap().offset();
    let mut new_vals: Vec<_> = capture_inds.iter().map(|i| self.stack[offset + i].shift()).collect();

    // Use the def that we just pushed. And double-shift to avoid borrow issues.
    let (_fn_def, cap_vals) = self.stack.last_mut().unwrap().as_fn_mut();
    cap_vals.clear();
    cap_vals.extend(new_vals.iter_mut().map(|v| v.shift()));
  }

  fn call(&mut self, morph_ind: usize, argc: usize, pos: &Position) -> FrameOp {
    // incr the IP here, since Opcode::Call returns from the op loop early and skips the normal incr(1).
    self.incr_ip(1);
    let fn_ind = self.stack.len() - argc - 1;

    // Arguments are on the stack before the call, let's add the captures here.
    self.push_captures(fn_ind);

    FrameOp::Push(Frame::new(fn_ind, morph_ind, true, pos.clone()))
  }

  fn push_captures(&mut self, fn_ind: usize) {
    let (_fn_def, capts) = match &mut self.stack[fn_ind] {
      Value::FnDef(f, c) => (f, c),

      // We don't push captures onto the stack for natives, instead we'll just pass them to the native function
      // directly.
      Value::NativeDef(..) => return,

      other => panic!("Not a function: {:?}", other)
    };

    let capts: Vec<_> = capts.iter_mut().map(|v| v.shift()).collect();

    for mut v in capts {
      self.stack_push(v.shift());
    }
  }

  fn ip(&self) -> usize { self.frame().ip() }
  fn incr_ip(&mut self, offset: usize) { self.frame_mut().incr_ip(offset); }
  fn frame(&self) -> &Frame { self.frames.last().unwrap() }
  fn frame_mut(&mut self) -> &mut Frame { self.frames.last_mut().unwrap() }
  fn stack_push(&mut self, val: Value<C>) { self.stack.push(val); }
  fn stack_pop(&mut self) -> Value<C> { self.stack.pop().unwrap() }

  fn stack_replace<F: FnOnce(&Value<C>) -> Value<C>>(&mut self, f: F) {
    // TODO(performance): replace the value inline on the stack
    let last = self.stack_pop();
    self.stack_push(f(&last))
  }

  fn debug_handle(&self, opcode: &Opcode) {
    if enabled!(Level::DEBUG) {
      println!("Stack: {}\nExecuting {:?} at {}", format_stack(&self.stack).as_str(), opcode, self.ip());
    }
  }
}

pub fn format_stack<C: Custom>(stack: &Stack<C>) -> String {
  let mut result = String::new();
  for v in stack {
    write(&mut result, format_args!("\n  {v:?}")).unwrap();
  }
  result
}

/// A call frame represents a current in-code call being executed.
///
/// A frame contains the return IP address, the stack offset, the current function's monomorph index, an
/// indicator if the VM should continue processing after the frame completes, and the code position of the call.
#[derive(Debug)]
pub struct Frame {
  ip: usize,
  offset: usize,
  morph_index: usize,
  going: bool,
  pos: Position
}

impl Frame {
  pub fn new(offset: usize, morph_index: usize, going: bool, pos: Position) -> Frame {
    Frame { ip: 0, offset, morph_index, going, pos }
  }

  pub fn pos(&self) -> &Position { &self.pos }
  pub fn going(&self) -> bool { self.going }
  pub fn ip(&self) -> usize { self.ip }
  pub fn incr_ip(&mut self, offset: usize) { self.ip += offset; }
  pub fn offset(&self) -> usize { self.offset }
  pub fn morph_index(&self) -> usize { self.morph_index }
}

pub enum FrameOp {
  Push(Frame),
  Pop
}

impl FrameOp {
  /// Perform the operation, and return if we should continue running.
  fn operate(self, frames: &mut Frames) -> bool {
    match self {
      Self::Push(frame) => {
        debug!("Pushing frame {:?}", frame);
        frames.push(frame);
        true
      }
      Self::Pop => {
        debug!("Popping frame");
        let frame = frames.pop().unwrap();
        frame.going()
      }
    }
  }
}

fn binary<C: Custom, F: FnOnce(&Value<C>, &Value<C>) -> Value<C>>(stack: &mut Stack<C>, f: F) {
  let last = stack.len() - 1;
  *stack.get_mut(last - 1).unwrap() = f(stack.get(last - 1).unwrap(), stack.get(last).unwrap());
  drop(stack.pop());
}
