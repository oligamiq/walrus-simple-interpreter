//! A tiny and incomplete Wasm interpreter
//!
//! This code's base is
//! https://github.com/rustwasm/wasm-bindgen/blob/c35cc9369d5e0dc418986f7811a0dd702fb33ef9/crates/wasm-interpreter/src/lib.rs
//!
//! This module contains a tiny and incomplete Wasm interpreter built on top of
//! `walrus`'s module structure. Each `Interpreter` contains some state
//! about the execution of a Wasm instance.

#![deny(missing_docs)]

use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, HashMap};
use walrus::ir::*;
use walrus::*;

/// A ready-to-go interpreter of a Wasm module.
///
/// An interpreter currently represents effectively cached state. It is reused
/// between calls to `interpret` and is precomputed from a `Module`. It houses
/// state like the Wasm stack, Wasm memory, etc.
pub struct Interpreter {
    /// The globals of the module. These are set up at the start of the
    /// module and are used to store the state of the module.
    pub globals: BTreeMap<GlobalId, Value>,
    /// The memory of the module. This is set up at the start of the
    /// module and is used to store the state of the module.
    pub mem: BTreeMap<MemoryId, BTreeMap<usize, [u8; 4]>>,
    /// The stack of the module. This is set up at the start of the
    /// module and is used to store the state of the module.
    pub stack: Vec<Value>,
    /// The functions of the module. These are set up at the start of the
    /// module and are used to store the state of the module.
    functions: HashMap<String, Box<dyn FnMut(&mut Interpreter, &[Value]) -> Result<Option<Value>>>>,
    /// interrupt handler eval
    interrupt_handler: Option<
        Box<dyn FnMut(&mut Interpreter, &Instr, (FunctionId, InstrSeqId, usize)) -> Result<()>>,
    >,
    init_memories: BTreeMap<MemoryId, Box<[(usize, Box<[u8]>)]>>,
}

impl Interpreter {
    /// Create a new interpreter for a given module.
    pub fn new(module: &Module) -> Result<Self> {
        let mut globals = BTreeMap::new();
        module
            .globals
            .iter()
            .map(|global| {
                if let GlobalKind::Local(ConstExpr::Value(v)) = global.kind {
                    globals.insert(global.id(), v);
                    Ok(())
                } else {
                    bail!("global is not a local constant");
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let init_memories = module
            .data
            .iter()
            .filter_map(|data| match data.kind {
                walrus::DataKind::Active { memory, offset } => {
                    if let ConstExpr::Value(v) = offset {
                        if let ir::Value::I32(offset) = v {
                            Some((
                                memory,
                                (offset as usize, data.value.clone().into_boxed_slice()),
                            ))
                        } else {
                            log::warn!("Data segment {:?} is not i32", offset);
                            None
                        }
                    } else {
                        log::warn!("Data segment {:?} is not a value", offset);
                        None
                    }
                }
                _ => {
                    log::warn!("Data segment passive is not supported");
                    None
                }
            })
            .fold(BTreeMap::new(), |mut acc, (memory, (offset, value))| {
                acc.entry(memory)
                    .or_insert_with(Vec::new)
                    .push((offset, value));
                acc
            })
            .into_iter()
            .map(|(memory, values)| (memory, values.into_boxed_slice()))
            .collect::<BTreeMap<_, _>>();

        let mem = module
            .memories
            .iter()
            .map(|mem| (mem.id(), BTreeMap::new()))
            .collect::<BTreeMap<_, _>>();

        Ok(Self {
            globals,
            mem,
            stack: Vec::with_capacity(256),
            functions: HashMap::new(),
            interrupt_handler: None,
            init_memories,
        })
    }

    /// Add a function to the interpreter. This function will be called
    /// whenever the interpreter encounters a call to it.
    pub fn add_function(
        &mut self,
        name: impl AsRef<str>,
        func: impl FnMut(&mut Interpreter, &[Value]) -> Result<Option<Value>> + 'static,
    ) {
        self.functions
            .insert(name.as_ref().to_string(), Box::new(func));
    }

    /// Add a function to the interpreter. This function will be called
    pub fn set_interrupt_handler(
        &mut self,
        handler: impl FnMut(&mut Interpreter, &Instr, (FunctionId, InstrSeqId, usize)) -> Result<()>
        + 'static,
    ) {
        self.interrupt_handler = Some(Box::new(handler));
    }

    /// Call the interrupt handler. This will call the interrupt handler
    pub fn call_interrupt_handler(
        &mut self,
        instr: &Instr,
        id: (FunctionId, InstrSeqId, usize),
    ) -> Result<()> {
        let mut interrupt_handler = self.interrupt_handler.take();
        if let Some(ref mut handler) = interrupt_handler {
            handler(self, instr, id)?;
        } else {
            bail!("no interrupt handler set");
        }

        self.interrupt_handler = interrupt_handler;

        Ok(())
    }

    fn mem_get(&mut self, id: MemoryId, address: u32) -> [u8; 4] {
        let address = address as usize;

        let mem = self.mem.get_mut(&id).unwrap();
        *mem.entry(address).or_insert_with(|| {
            self.init_memories
                .get(&id)
                .unwrap()
                .iter()
                .find(|(offset, data)| offset <= &address && address < offset + data.len())
                .map(|(offset, value)| {
                    let offset = address - offset;
                    let mut arr = [0; 4];
                    arr.copy_from_slice(&value[offset..offset + 4]);
                    arr
                })
                .unwrap_or([0; 4])
        })
    }

    fn mem_set(&mut self, id: MemoryId, address: u32, value: [u8; 4]) {
        let address = address as usize;
        let mem = self.mem.get_mut(&id).unwrap();
        mem.insert(address, value);
    }

    fn stack_pop(&mut self) -> Result<Value> {
        self.stack.pop().with_context(|| "stack underflow")
    }

    fn stack_push(&mut self, value: Value) {
        self.stack.push(value);
    }

    fn stack_tee(&self) -> Value {
        self.stack.last().cloned().unwrap()
    }

    /// Call a function in the module with the given arguments.
    pub fn call(
        &mut self,
        id: FunctionId,
        module: &Module,
        args: &[Value],
    ) -> Result<Option<Value>> {
        let func = module.funcs.get(id);
        log::debug!("starting a call of {:?} {:?}", id, func.name);
        log::debug!("arguments {:?}", args);
        let local = match &func.kind {
            walrus::FunctionKind::Local(l) => l,
            walrus::FunctionKind::Import(import) => {
                let id = import.import;
                let func = module.imports.get(id);
                let name = func.name.clone();
                let mut func = self
                    .functions
                    .remove(&name)
                    .with_context(|| format!("function {name} not found"))?;

                let ret = func(self, args);

                self.functions.insert(name, func);

                return ret;
            }
            _ => bail!("function is not local"),
        };

        let entry = local.entry_block();
        let block = local.block(entry);

        let mut locals = BTreeMap::new();

        let mut frame = Frame {
            module,
            interp: self,
            locals: &mut locals,
            done: false,
            local_func: local,
        };

        assert_eq!(local.args.len(), args.len());
        for (arg, val) in local.args.iter().zip(args) {
            frame.locals.insert(*arg, *val);
        }

        for (i, (instr, _)) in block.instrs.iter().enumerate() {
            if let Err(err) = frame.eval(instr, (id, entry, i)) {
                if let Some(name) = &module.funcs.get(id).name {
                    bail!("{name}: {err}")
                } else {
                    bail!("{err}")
                }
            }

            if frame.done {
                break;
            }
        }
        Ok(self.stack.last().cloned())
    }
}

struct Frame<'a, 'b> {
    module: &'a Module,
    interp: &'a mut Interpreter,
    local_func: &'a LocalFunction,
    locals: &'b mut BTreeMap<LocalId, Value>,
    done: bool,
}

impl<'a, 'b> Frame<'a, 'b> {
    fn eval(&mut self, instr: &Instr, id: (FunctionId, InstrSeqId, usize)) -> Result<()> {
        use walrus::ir::*;

        self.interp.call_interrupt_handler(instr, id)?;

        match instr {
            Instr::Const(Const { value }) => self.interp.stack_push(*value),
            Instr::LocalGet(e) => self
                .interp
                .stack_push(self.locals.get(&e.local).cloned().unwrap()),
            Instr::LocalSet(e) => {
                let val = self.interp.stack_pop()?;
                self.locals.insert(e.local, val);
            }
            Instr::LocalTee(e) => {
                let val = self.interp.stack_tee();
                self.locals.insert(e.local, val);
            }

            Instr::GlobalGet(e) => self
                .interp
                .stack_push(self.interp.globals.get(&e.global).cloned().unwrap()),
            Instr::GlobalSet(e) => {
                let val = self.interp.stack_pop()?;
                *self.interp.globals.get_mut(&e.global).unwrap() = val;
            }

            // Support simple arithmetic, mainly for the stack pointer
            // manipulation
            Instr::Binop(e) => {
                let rhs = self.interp.stack_pop()?;
                let lhs = self.interp.stack_pop()?;
                match (rhs, lhs) {
                    (Value::I32(rhs), Value::I32(lhs)) => {
                        self.interp.stack_push(Value::I32(match e.op {
                            BinaryOp::I32Sub => lhs - rhs,
                            BinaryOp::I32Add => lhs + rhs,
                            BinaryOp::I32Mul => lhs * rhs,
                            BinaryOp::I32DivU => lhs / rhs,
                            BinaryOp::I32DivS => lhs / rhs,
                            BinaryOp::I32RemU => lhs % rhs,
                            BinaryOp::I32RemS => lhs % rhs,
                            BinaryOp::I32And => lhs & rhs,
                            BinaryOp::I32Or => lhs | rhs,
                            BinaryOp::I32Xor => lhs ^ rhs,
                            BinaryOp::I32Shl => lhs << rhs,
                            BinaryOp::I32ShrU => lhs >> rhs,
                            BinaryOp::I32ShrS => lhs >> rhs,
                            BinaryOp::I32Rotl => lhs.rotate_left(rhs as u32),
                            BinaryOp::I32Rotr => lhs.rotate_right(rhs as u32),
                            BinaryOp::I32Eq => (lhs == rhs) as i32,
                            BinaryOp::I32Ne => (lhs != rhs) as i32,
                            BinaryOp::I32LtU => (lhs < rhs) as i32,
                            BinaryOp::I32LtS => (lhs < rhs) as i32,
                            BinaryOp::I32GtU => (lhs > rhs) as i32,
                            BinaryOp::I32GtS => (lhs > rhs) as i32,
                            BinaryOp::I32LeU => (lhs <= rhs) as i32,
                            BinaryOp::I32LeS => (lhs <= rhs) as i32,
                            BinaryOp::I32GeU => (lhs >= rhs) as i32,
                            BinaryOp::I32GeS => (lhs >= rhs) as i32,
                            op => bail!("invalid binary op {:?}", op),
                        }));
                    }
                    (Value::I64(rhs), Value::I64(lhs)) => {
                        self.interp.stack_push(Value::I64(match e.op {
                            BinaryOp::I64Sub => lhs - rhs,
                            BinaryOp::I64Add => lhs + rhs,
                            op => bail!("invalid binary op {:?}", op),
                        }));
                    }
                    _ => bail!("invalid types for binary op"),
                }
            }

            Instr::Unop(e) => {
                let val = self.interp.stack_pop()?;
                match val {
                    Value::I32(val) => {
                        self.interp.stack_push(Value::I32(match e.op {
                            UnaryOp::I32Clz => val.leading_zeros() as i32,
                            UnaryOp::I32Ctz => val.trailing_zeros() as i32,
                            UnaryOp::I32Popcnt => val.count_ones() as i32,
                            UnaryOp::I32Eqz => (val == 0) as i32,
                            op => bail!("invalid unary op {:?}", op),
                        }));
                    }
                    _ => bail!("invalid types for unary op"),
                }
            }

            // Support small loads/stores to the stack. These show up in debug
            // mode where there's some traffic on the linear stack even when in
            // theory there doesn't need to be.
            Instr::Load(e) => {
                let address = self.interp.stack_pop()?;
                let address = if let Value::I32(address) = address {
                    address
                } else {
                    bail!("invalid address type for load");
                };
                ensure!(
                    address > 0,
                    "Read a negative address value from the stack. Did we run out of memory?"
                );
                let address = address as u32 + e.arg.offset;
                ensure!(address % 4 == 0);
                let value = self.interp.mem_get(e.memory, address);
                let value = match e.kind {
                    LoadKind::I32 { .. } => Value::I32(u32::from_le_bytes(value) as i32),
                    LoadKind::I64 { .. } => {
                        let next_value = self.interp.mem_get(e.memory, address + 4);
                        let value =
                            u64::from_le_bytes([value, next_value].concat().try_into().unwrap());
                        Value::I64(value as i64)
                    }
                    LoadKind::F32 => todo!(),
                    LoadKind::F64 => todo!(),
                    _ => bail!("no support for this load kind"),
                };
                self.interp.stack_push(value);
            }
            Instr::Store(e) => {
                let value = self.interp.stack_pop()?;
                let address = self.interp.stack_pop()?;
                let address = if let Value::I32(address) = address {
                    address
                } else {
                    bail!("invalid address type for load");
                };
                ensure!(
                    address > 0,
                    "Read a negative address value from the stack. Did we run out of memory?"
                );
                let address = address as u32 + e.arg.offset;
                ensure!(address % 4 == 0);
                match e.kind {
                    StoreKind::I32 { .. } => {
                        let value = if let Value::I32(value) = value {
                            value
                        } else {
                            bail!("invalid value type for store");
                        };
                        let v = u32::to_le_bytes(value as u32);
                        self.interp.mem_set(e.memory, address, v);
                    }
                    StoreKind::I64 { .. } => {
                        let value = if let Value::I64(value) = value {
                            value
                        } else {
                            bail!("invalid value type for store");
                        };
                        let value = u64::to_le_bytes(value as u64);
                        let v = value.chunks(4).map(|chunk| {
                            let mut arr = [0; 4];
                            arr.copy_from_slice(chunk);
                            arr
                        });
                        let mut i = 0;
                        for chunk in v {
                            self.interp.mem_set(e.memory, address + i, chunk);
                            i += 4;
                        }
                    }
                    StoreKind::F32 => todo!(),
                    StoreKind::F64 => todo!(),
                    _ => bail!("no support for this store kind"),
                };
            }

            Instr::Return(_) => {
                log::debug!("return");
                self.done = true;
            }

            Instr::Drop(_) => {
                log::debug!("drop");
                self.interp.stack_pop()?;
            }

            Instr::Call(Call { func }) | Instr::ReturnCall(ReturnCall { func }) => {
                let func = *func;

                let ty = self.module.types.get(self.module.funcs.get(func).ty());
                let args = (0..ty.params().len())
                    .map(|_| self.interp.stack_pop())
                    .collect::<Result<Vec<_>>>()?;

                let ret = self.interp.call(func, self.module, &args)?;
                if let Some(ret) = ret {
                    self.interp.stack_push(ret);
                }
            }

            Instr::Block(b) => {
                log::debug!("block");

                self.block(id.0, b.seq)?;
            }

            Instr::Loop(l) => {
                log::debug!("loop");
                self.block(id.0, l.seq)?;
            }

            Instr::BrIf(i) => {
                log::debug!("br_if");
                let val = self.interp.stack_pop()?;
                let val = if let Value::I32(val) = val {
                    val
                } else {
                    bail!("invalid value type for br_if");
                };
                if val != 0 {
                    self.block(id.0, i.block)?;
                }
            }

            Instr::Br(i) => {
                log::debug!("br");
                self.block(id.0, i.block)?;
            }

            Instr::IfElse(i) => {
                log::debug!("if_else");
                let val = self.interp.stack_pop()?;
                let val = if let Value::I32(val) = val {
                    val
                } else {
                    bail!("invalid value type for if_else");
                };
                if val != 0 {
                    self.block(id.0, i.consequent)?;
                } else {
                    self.block(id.0, i.alternative)?;
                }
            }

            // All other instructions shouldn't be used by our various
            // descriptor functions. LLVM optimizations may mean that some
            // of the above instructions aren't actually needed either, but
            // the above instructions have empirically been required when
            // executing our own test suite in wasm-bindgen.
            //
            // Note that LLVM may change over time to generate new
            // instructions in debug mode, and we'll have to react to those
            // sorts of changes as they arise.
            s => bail!("unknown instruction {:?}", s),
        }

        Ok(())
    }

    fn block(&mut self, function_id: FunctionId, instr_sec_id: InstrSeqId) -> Result<()> {
        let block = self.local_func.block(instr_sec_id);
        for (i, (instr, _)) in block.instrs.iter().enumerate() {
            self.eval(instr, (function_id, instr_sec_id, i))?;
            if self.done {
                return Ok(());
            }
        }
        Ok(())
    }
}
