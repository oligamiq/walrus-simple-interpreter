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
    /// The tables of the module. These are set up at the start of the
    /// module and are used to store the state of the module.
    pub tables: BTreeMap<TableId, Vec<FunctionId>>,
    /// The functions of the module. These are set up at the start of the
    /// module and are used to store the state of the module.
    functions: HashMap<String, Box<dyn FnMut(&mut Interpreter, &[Value]) -> Result<Vec<Value>>>>,
    /// interrupt handler eval
    interrupt_handler: Option<
        Box<dyn FnMut(&mut Interpreter, &Instr, (FunctionId, InstrSeqId, usize)) -> Result<()>>,
    >,
    /// interrupt handler mem access
    /// (&mut interpreter, &instr, (function_id, instr_seq_id, i), (memory_id, address, value, access_type))
    /// Called after the change
    interrupt_handler_mem: Option<
        Box<
            dyn FnMut(
                &mut Interpreter,
                &Instr,
                (FunctionId, InstrSeqId, usize),
                (MemoryId, u32, Value, MemoryAccessType),
            ) -> Result<()>,
        >,
    >,

    init_memories: BTreeMap<MemoryId, Box<[(usize, Box<[u8]>)]>>,

    /// The size of the memory of the module. This is set up at the start of
    /// the module and is used to store the state of the module.
    /// pages per 64kiB
    pub memory_size: BTreeMap<MemoryId, usize>,
}

/// The type of memory access.
pub enum MemoryAccessType {
    /// Memory load access.
    Load,
    /// Memory store access.
    Store,
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

        let memory_size = module
            .memories
            .iter()
            .map(|mem| (mem.id(), mem.initial as usize))
            .collect::<BTreeMap<_, _>>();

        let tables = module
            .tables
            .iter()
            .map(|table| {
                (table.id(), {
                    let mut vec = Vec::with_capacity(table.initial as usize);
                    for elem_id in &table.elem_segments {
                        let elem = module.elements.get(*elem_id);
                        match elem.kind {
                            walrus::ElementKind::Active { offset, .. } => {
                                let offset = match offset {
                                    ConstExpr::Value(ir::Value::I32(offset)) => offset as usize,
                                    _ => {
                                        log::warn!("Table segment {:?} is not i32", offset);
                                        continue;
                                    }
                                };
                                let items = match &elem.items {
                                    ElementItems::Functions(ids) => ids,
                                    ElementItems::Expressions(..) => todo!(),
                                };

                                if offset > vec.len() {
                                    vec.resize(offset, items[0]);
                                } else if offset < vec.len() {
                                    vec.truncate(offset);
                                }
                                vec.extend(items);
                            }
                            walrus::ElementKind::Passive => {
                                log::warn!("Table segment {:?} is passive", elem_id);
                            }
                            ElementKind::Declared => {
                                todo!("Declared table segments are not supported")
                            }
                        }
                    }
                    vec
                })
            })
            .collect::<BTreeMap<_, _>>();

        Ok(Self {
            globals,
            mem,
            stack: Vec::with_capacity(256),
            functions: HashMap::new(),
            tables,
            interrupt_handler: None,
            interrupt_handler_mem: None,
            init_memories,
            memory_size,
        })
    }

    /// Set the value
    pub fn mem_set_i32(&mut self, id: MemoryId, address: u32, value: i32) -> Result<()> {
        let value = u32::to_le_bytes(value as u32);
        Ok(self.mem_set(id, address, value))
    }

    /// Add a function to the interpreter. This function will be called
    /// whenever the interpreter encounters a call to it.
    pub fn add_function(
        &mut self,
        name: impl AsRef<str>,
        func: impl FnMut(&mut Interpreter, &[Value]) -> Result<Vec<Value>> + 'static,
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

    /// Add a function to the interpreter. This function will be called
    /// whenever the interpreter encounters a call to it.
    pub fn set_interrupt_handler_mem(
        &mut self,
        handler: impl FnMut(
            &mut Interpreter,
            &Instr,
            (FunctionId, InstrSeqId, usize),
            (MemoryId, u32, Value, MemoryAccessType),
        ) -> Result<()>
        + 'static,
    ) {
        self.interrupt_handler_mem = Some(Box::new(handler));
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
        }

        self.interrupt_handler = interrupt_handler;

        Ok(())
    }

    /// Call the interrupt handler. This will call the interrupt handler
    pub fn call_interrupt_handler_mem(
        &mut self,
        instr: &Instr,
        id: (FunctionId, InstrSeqId, usize),
        mem: (MemoryId, u32, Value, MemoryAccessType),
    ) -> Result<()> {
        let mut interrupt_handler = self.interrupt_handler_mem.take();
        if let Some(ref mut handler) = interrupt_handler {
            handler(self, instr, id, mem)?;
        }

        self.interrupt_handler_mem = interrupt_handler;

        Ok(())
    }

    #[allow(unused)]
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

    #[allow(unused)]
    fn mem_set(&mut self, id: MemoryId, address: u32, value: [u8; 4]) {
        let address = address as usize;
        let mem = self.mem.get_mut(&id).unwrap();
        mem.insert(address, value);
    }

    #[allow(unused)]
    fn mem_set_u8(&mut self, id: MemoryId, address: u32, value: u8) {
        let address = address as usize;
        let mem = self.mem.get_mut(&id).unwrap();
        if let Some(entry) = mem.get_mut(&address) {
            entry[0] = value;
        } else {
            let mut entry = [0; 4];
            entry[0] = value;
            mem.insert(address, entry);
        }
    }

    #[allow(unused)]
    fn stack_pop(&mut self) -> Result<Value> {
        self.stack.pop().with_context(|| "stack underflow")
    }

    #[allow(unused)]
    fn stack_push(&mut self, value: Value) {
        self.stack.push(value);
    }

    #[allow(unused)]
    fn stack_extend(&mut self, values: Vec<Value>) {
        self.stack.extend(values);
    }

    #[allow(unused)]
    fn stack_tee(&self) -> Value {
        self.stack.last().cloned().unwrap()
    }

    /// Call a function in the module with the given arguments.
    pub fn call(&mut self, id: FunctionId, module: &Module, args: &[Value]) -> Result<Vec<Value>> {
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

        assert_eq!(local.args.len(), args.len());
        let mut locals = BTreeMap::new();
        for (arg, val) in local.args.iter().zip(args) {
            locals.insert(*arg, *val);
        }

        let mut frame = Frame {
            module,
            interp: self,
            locals,
            local_func: local,
        };

        for (i, (instr, _)) in block.instrs.iter().enumerate() {
            match frame.eval(instr, (id, entry, i)) {
                Ok(BlockRet::Success) => {}
                Ok(BlockRet::Break { .. }) => unreachable!(),
                Ok(BlockRet::Return) => {
                    let ty = match block.ty {
                        InstrSeqType::Simple(val_type) => {
                            val_type.map(|v| vec![v]).unwrap_or_default()
                        }
                        InstrSeqType::MultiValue(id) => {
                            let ty = module.types.get(id);
                            ty.results().iter().map(|v| *v).collect()
                        }
                    };
                    let ret = self.stack.split_off(self.stack.len() - ty.len());
                    self.stack.clear();

                    return Ok(ret);
                }
                Err(err) => {
                    if let Some(name) = &module.funcs.get(id).name {
                        bail!("{name}: {err}")
                    } else {
                        bail!("{err}")
                    }
                }
            }
        }
        Ok(self.stack.clone())
    }
}

struct Frame<'a> {
    module: &'a Module,
    interp: &'a mut Interpreter,
    local_func: &'a LocalFunction,
    locals: BTreeMap<LocalId, Value>,
}

macro_rules! block {
    (
        $self:ident,
        $id:ident,
        $seq:expr
    ) => {{
        return $self.block($id, $seq);
    }};
}

impl Frame<'_> {
    fn eval(&mut self, instr: &Instr, place: (FunctionId, InstrSeqId, usize)) -> Result<BlockRet> {
        use walrus::ir::*;

        fn as_u32(v: i32) -> u32 {
            #[allow(unnecessary_transmutes)]
            unsafe {
                core::mem::transmute(v)
            }
        }

        fn as_i32(v: u32) -> i32 {
            #[allow(unnecessary_transmutes)]
            unsafe {
                core::mem::transmute(v)
            }
        }

        let id = place.0;

        self.interp.call_interrupt_handler(instr, place)?;

        match instr {
            Instr::Const(Const { value }) => self.interp.stack_push(*value),
            Instr::LocalGet(e) => {
                self.interp
                    .stack_push(self.locals.get(&e.local).cloned().unwrap_or_else(|| {
                        let ty = self.module.locals.get(e.local).ty();
                        match ty {
                            ValType::I32 => Value::I32(0),
                            ValType::I64 => Value::I64(0),
                            ValType::F32 => Value::F32(0.0),
                            ValType::F64 => Value::F64(0.0),
                            _ => unreachable!(),
                        }
                    }))
            }

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
                            BinaryOp::I32DivU => as_i32(as_u32(lhs) / as_u32(rhs)),
                            BinaryOp::I32DivS => lhs / rhs,
                            BinaryOp::I32RemU => as_i32(as_u32(lhs) % as_u32(rhs)),
                            BinaryOp::I32RemS => lhs % rhs,
                            BinaryOp::I32And => lhs & rhs,
                            BinaryOp::I32Or => lhs | rhs,
                            BinaryOp::I32Xor => lhs ^ rhs,
                            BinaryOp::I32Shl => lhs << rhs,
                            BinaryOp::I32ShrU => as_i32(as_u32(lhs) >> as_u32(rhs)),
                            BinaryOp::I32ShrS => lhs >> rhs,
                            BinaryOp::I32Rotl => lhs.rotate_left(rhs as u32),
                            BinaryOp::I32Rotr => lhs.rotate_right(rhs as u32),
                            BinaryOp::I32Eq => (lhs == rhs) as i32,
                            BinaryOp::I32Ne => (lhs != rhs) as i32,
                            BinaryOp::I32LtU => (as_u32(lhs) < as_u32(rhs)) as i32,
                            BinaryOp::I32LtS => (lhs < rhs) as i32,
                            BinaryOp::I32GtU => (as_u32(lhs) > as_u32(rhs)) as i32,
                            BinaryOp::I32GtS => (lhs > rhs) as i32,
                            BinaryOp::I32LeU => (as_u32(lhs) <= as_u32(rhs)) as i32,
                            BinaryOp::I32LeS => (lhs <= rhs) as i32,
                            BinaryOp::I32GeU => (as_u32(lhs) >= as_u32(rhs)) as i32,
                            BinaryOp::I32GeS => (lhs >= rhs) as i32,
                            op => bail!("invalid binary op {:?}", op),
                        }));
                    }
                    (Value::I64(rhs), Value::I64(lhs)) => match e.op {
                        BinaryOp::I64Sub => self.interp.stack_push(Value::I64(lhs - rhs)),
                        BinaryOp::I64Add => self.interp.stack_push(Value::I64(lhs + rhs)),
                        BinaryOp::I64Eq => {
                            self.interp.stack_push(Value::I32((lhs == rhs) as i32));
                        }
                        BinaryOp::I64Or => self.interp.stack_push(Value::I64(lhs | rhs)),
                        op => bail!("invalid binary op {:?}", op),
                    },
                    v => bail!("invalid types for binary op: {v:?}\ne: {e:?}"),
                }
            }

            Instr::Unop(e) => {
                let val = self.interp.stack_pop()?;
                match val {
                    Value::I32(val) => {
                        self.interp.stack_push(match e.op {
                            UnaryOp::I32Clz => Value::I32(val.leading_zeros() as i32),
                            UnaryOp::I32Ctz => Value::I32(val.trailing_zeros() as i32),
                            UnaryOp::I32Popcnt => Value::I32(val.count_ones() as i32),
                            UnaryOp::I32Eqz => Value::I32((val == 0) as i32),
                            // The extend instructions, are used to convert (extend) numbers of type i32 to type i64.
                            // There are signed and unsigned versions of this instruction.
                            UnaryOp::I64ExtendUI32 => Value::I64((val as u32) as i64),
                            op => bail!("invalid unary op {:?}", op),
                        });
                    }
                    Value::I64(val) => {
                        self.interp.stack_push(Value::I32(match e.op {
                            UnaryOp::I64Eqz => (val == 0) as i32,
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
                let address = address as u32 + e.arg.offset;
                ensure!(
                    address > 0,
                    "Read a negative address value from the stack. Did we run out of memory?"
                );
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
                    LoadKind::I32_8 { kind } => {
                        let value = i32::from_le_bytes([value[0], 0, 0, 0]);
                        let value = match kind {
                            ExtendedLoad::SignExtend => value,
                            ExtendedLoad::ZeroExtend => value as u32 as i32,
                            ExtendedLoad::ZeroExtendAtomic => value as u32 as i32,
                        };
                        Value::I32(value)
                    }
                    variant => bail!("no support for this load kind: {variant:?}"),
                };

                self.interp.call_interrupt_handler_mem(
                    instr,
                    place,
                    (e.memory, address, value, MemoryAccessType::Load),
                )?;
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
                let address = address as u32 + e.arg.offset;
                ensure!(
                    address > 0,
                    "Read a negative address value from the stack. Did we run out of memory?"
                );
                ensure!(address % 4 == 0);
                match e.kind {
                    StoreKind::I32 { .. } => {
                        let value = if let Value::I32(value) = value {
                            value
                        } else {
                            bail!("invalid value type for store");
                        };
                        let v = u32::to_le_bytes(value as u32);
                        self.interp.call_interrupt_handler_mem(
                            instr,
                            place,
                            (
                                e.memory,
                                address,
                                Value::I32(value),
                                MemoryAccessType::Store,
                            ),
                        )?;
                        self.interp.mem_set(e.memory, address, v);
                    }
                    // store a 32-bit number into an 8-bit slot in memory using i32.store8
                    // If the number doesn't fit in the narrower number type it will wrap.
                    StoreKind::I32_8 { .. } => {
                        let value = if let Value::I32(value) = value {
                            value
                        } else {
                            bail!("invalid value type for store");
                        };
                        let v = u32::to_le_bytes(value as u32)[0];
                        self.interp.call_interrupt_handler_mem(
                            instr,
                            place,
                            (
                                e.memory,
                                address,
                                Value::I32(value),
                                MemoryAccessType::Store,
                            ),
                        )?;
                        self.interp.mem_set_u8(e.memory, address, v);
                    }
                    StoreKind::I64 { .. } => {
                        let v = if let Value::I64(value) = value {
                            value
                        } else {
                            bail!("invalid value type for store");
                        };
                        let v = u64::to_le_bytes(v as u64);
                        let v = v.chunks(4).map(|chunk| {
                            let mut arr = [0; 4];
                            arr.copy_from_slice(chunk);
                            arr
                        });
                        let mut i = 0;
                        for chunk in v {
                            self.interp.mem_set(e.memory, address + i, chunk);
                            i += 4;
                        }
                        self.interp.call_interrupt_handler_mem(
                            instr,
                            place,
                            (e.memory, address, value, MemoryAccessType::Store),
                        )?;
                    }
                    StoreKind::F32 => todo!(),
                    StoreKind::F64 => todo!(),
                    v => bail!("no support for this store kind: {v:?}"),
                };
            }

            Instr::Return(_) => {
                log::debug!("return");

                return Ok(BlockRet::Return);
            }

            Instr::Drop(_) => {
                log::debug!("drop");
                self.interp.stack_pop()?;
            }

            Instr::Call(Call { func }) => {
                let func = *func;

                let ty = self.module.types.get(self.module.funcs.get(func).ty());

                let args = self
                    .interp
                    .stack
                    .split_off(self.interp.stack.len() - ty.params().len());

                let ret = self.interp.call(func, self.module, &args)?;
                self.interp.stack.extend(ret);
            }

            Instr::CallIndirect(CallIndirect {
                ty,
                table: table_id,
            }) => {
                log::debug!("call_indirect");

                let func_index = self.interp.stack_pop()?;
                let table = self.interp.tables.get(table_id).unwrap();
                let func_index = if let Value::I32(func_index) = func_index {
                    func_index as usize
                } else {
                    bail!("invalid function index type for call_indirect");
                };

                ensure!(
                    func_index < table.len(),
                    "function index out of bounds for call_indirect"
                );

                let func_id = table[func_index];
                let ty = self.module.types.get(*ty);
                let args = self
                    .interp
                    .stack
                    .split_off(self.interp.stack.len() - ty.params().len());

                let ret = self.interp.call(func_id, self.module, &args)?;
                self.interp.stack.extend(ret);
            }

            Instr::Block(b) => {
                log::debug!("block");

                block!(self, id, b.seq);
            }

            Instr::Loop(l) => {
                log::debug!("loop");

                loop {
                    match self.block(id, l.seq)? {
                        BlockRet::Success => return Ok(BlockRet::Success),
                        BlockRet::Break { break_point } => {
                            if break_point != l.seq {
                                return Ok(BlockRet::Break { break_point });
                            }
                        }
                        BlockRet::Return => return Ok(BlockRet::Return),
                    }
                }
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
                    return Ok(BlockRet::Break {
                        break_point: i.block,
                    });
                }
            }

            Instr::Br(i) => {
                log::debug!("br");
                return Ok(BlockRet::Break {
                    break_point: i.block,
                });
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
                    block!(self, id, i.consequent);
                } else {
                    block!(self, id, i.alternative);
                }
            }

            Instr::Select(i) => {
                log::debug!("select");

                if i.ty.is_some() {
                    todo!("select with type");
                }

                let val = self.interp.stack_pop()?;
                let val = if let Value::I32(val) = val {
                    val
                } else {
                    bail!("invalid value type for select");
                };
                let b = self.interp.stack_pop()?;
                let a = self.interp.stack_pop()?;
                if val != 0 {
                    self.interp.stack_push(a);
                } else {
                    self.interp.stack_push(b);
                }
            }

            Instr::MemorySize(i) => {
                log::debug!("memory size");
                let id = i.memory;
                let size = self.interp.memory_size.get(&id).unwrap();
                self.interp.stack_push(Value::I32(*size as i32));
            }

            Instr::MemoryGrow(i) => {
                log::debug!("memory grow");
                let id = i.memory;
                let size = *self.interp.memory_size.get(&id).unwrap();
                let val = self.interp.stack_pop()?;
                let val = if let Value::I32(val) = val {
                    val
                } else {
                    bail!("invalid value type for memory grow");
                };
                if val < 0 {
                    bail!("invalid value for memory grow");
                }
                self.interp.memory_size.insert(id, size + val as usize);
            }

            Instr::Unreachable(_) => {
                log::debug!("unreachable");
                bail!("unreachable");
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

        Ok(BlockRet::Success)
    }

    fn block(&mut self, function_id: FunctionId, instr_sec_id: InstrSeqId) -> Result<BlockRet> {
        let unwind_stack_height = self.interp.stack.len();
        let block = self.local_func.block(instr_sec_id);
        for (i, (instr, _)) in block.instrs.iter().enumerate() {
            let ret = self.eval(instr, (function_id, instr_sec_id, i))?;
            if let BlockRet::Break { break_point } = ret {
                if break_point == break_point {
                    let block_ty = match block.ty {
                        InstrSeqType::Simple(val_type) => {
                            val_type.map(|v| vec![v]).unwrap_or_default()
                        }
                        InstrSeqType::MultiValue(id) => {
                            let ty = self.module.types.get(id);
                            ty.results().iter().map(|v| *v).collect()
                        }
                    };

                    let ret = self
                        .interp
                        .stack
                        .split_off(self.interp.stack.len() - block_ty.len());

                    self.interp.stack.truncate(unwind_stack_height);

                    self.interp.stack_extend(ret);

                    return Ok(BlockRet::Success);
                }

                return Ok(ret);
            }
        }
        Ok(BlockRet::Success)
    }
}

enum BlockRet {
    Success,
    Break { break_point: InstrSeqId },
    Return,
}
