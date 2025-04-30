# simple wasm interpreter
This dependents walrus, anyhow, log only.

- call func with virtual system per eval operation
- register import func
- Coverage of some instruction. If not enough, we recommend pull-request.
- only 500 code lines
- used by my [other project](https://github.com/oligamiq/wasip1-vfs)

```rust
let mut interpreter = walrus_simple_interpreter::Interpreter::new(&module)?;
interpreter.set_interrupt_handler(|_, instr, _| {
    println!("Interrupt handler called");
    println!("Instr: {instr:?}");

    Ok(())
});
interpreter.call(fid, self, &[])?;
```
