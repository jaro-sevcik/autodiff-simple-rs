# Autodiff in Rust

This is an experiment to rebuild some parts of autodiff in Rust.

In particular, I have been inspired by JAX
[autodidax](https://jax.readthedocs.io/en/latest/autodidax.html).
We are only looking to replicate some specific aspect of autodidax.

### Goals
* Enable using one generic (single float input, single float output) function 
  for evaluation, compilation and gradient.
* Use tracing and tracers to interpret the expression in particular context
  (evaluation, grad, etc.).
* Ensure that the tracers are compositional:
  * Enable applying gradient multiple times.
  * Enable compiling gradient and computing gradient from compiled code.
* Implement forward and backward gradient computation.


### Stretch goals
* Extend to vectors or tensors.
* Enable structured parameter space (flattening of structures, JAX-style).
* Compilation to XLA.

### Super stretch goals
* Implement some simple neural net training algorithm.

### Unlikely goals
Here are some goals that we might but likely won't tackle:
* Integration with operator overloading.

### Non-goals
* Dynamic tracing.

## Notes, TODO

**Done**
* Simple f32 tracer interface.
* Simple eval tracing.
* Simple grad tracing.
* Composable grad tracing.
  * Generic grad tracer (tracer of a tracer).
  * Composable wrapper for grad.
* Simple tracing into expressions.
  * Composing grad with expression tracing.
* Backward gradient.
* Multiple inputs.
* Multiple outputs.
* Refactor into modules.
* Non-grad parameters.

**TODO**
* Simple tracing into expressions.
  * Caching for compiled expressions.
* Tensors.
  * Types: i32, f32
  * Ops:
    * Matmul (with grad).
    * Addition  (with grad).
    * tanh  (with grad).
    * Broadcast (with grad).
    * Logsumexp (with grad).
    * Reshape (with grad).
    * Indexing (with only the index grad).
* Flattening (likely inspired by something like serde).
* Compile expressions to XLA.
* Neural net example.

# Development

## Logging

Log all `trace!` invocations from the grad module:
```
RUST_LOG=autodiff_simple_rs::grad=trace cargo run
```