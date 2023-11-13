use std::cell::RefCell;
use std::sync::Arc;

#[derive(Debug, Clone)]
enum Primitive {
    Mul,
    Add,
    Constant(f32),
}

// Represents a trace.
// Contains factory methods for creating tracers.
trait Trace {
    type Value: Clone;
    fn primitive(&self, prim: Primitive, inputs: &[&Self::Value]) -> Self::Value;

    fn add(&self, lhs: &Self::Value, rhs: &Self::Value) -> Self::Value {
        self.primitive(Primitive::Add, &[lhs, rhs])
    }
    fn mul(&self, lhs: &Self::Value, rhs: &Self::Value) -> Self::Value {
        self.primitive(Primitive::Mul, &[lhs, rhs])
    }
    fn constant(&self, value: f32) -> Self::Value {
        self.primitive(Primitive::Constant(value), &[])
    }
}

#[derive(Debug, Clone)]
struct EvalTrace {}

impl Trace for EvalTrace {
    type Value = f32;

    fn primitive(&self, prim: Primitive, inputs: &[&Self::Value]) -> Self::Value {
        match prim {
            Primitive::Constant(c) => c,
            Primitive::Add => inputs[0] + inputs[1],
            Primitive::Mul => inputs[0] * inputs[1],
        }
    }
}

#[derive(Debug, Clone)]
struct GradTracer<T> {
    value: T,
    grad: T,
}

impl<T> GradTracer<T> {
    fn new(value: T, grad: T) -> Self {
        Self { value, grad }
    }
}

#[derive(Debug, Clone)]
struct GradTrace<Inner: Trace> {
    inner: Inner,
}

impl<Inner: Trace> Trace for GradTrace<Inner> {
    type Value = GradTracer<Inner::Value>;

    fn primitive(&self, prim: Primitive, inputs: &[&Self::Value]) -> Self::Value {
        match prim {
            Primitive::Constant(c) => {
                assert_eq!(inputs.len(), 0);
                Self::Value {
                    value: self.inner.constant(c),
                    grad: self.inner.constant(0.0),
                }
            }
            Primitive::Add => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.add(&inputs[0].value, &inputs[1].value);
                let grad = self.inner.add(&inputs[0].grad, &inputs[1].grad);
                Self::Value::new(value, grad)
            }
            Primitive::Mul => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.mul(&inputs[0].value, &inputs[1].value);
                let grad = self.inner.add(
                    &self.inner.mul(&inputs[0].value, &inputs[1].grad),
                    &self.inner.mul(&inputs[0].grad, &inputs[1].value),
                );
                Self::Value::new(value, grad)
            }
        }
    }
}

fn grad<T: Trace + Clone, GF>(fun: GF) -> impl Fn(&T, &T::Value) -> T::Value
where
    GF: Fn(&GradTrace<T>, &<GradTrace<T> as Trace>::Value) -> <GradTrace<T> as Trace>::Value,
{
    move |trace, value| {
        let grad_trace = GradTrace {
            inner: trace.clone(),
        };
        let value_for_grad = GradTracer::<T::Value>::new(value.clone(), trace.constant(1.0));
        let result = fun(&grad_trace, &value_for_grad);
        result.grad
    }
}

#[derive(Debug, Clone)]
struct ExprTracer {
    // TODO figure out what to do with the cyclic
    // reference. Perhaps thios should be a reference with
    // a lifetime at least as long as this object?
    trace: ExprTrace,
    value: TracedArgument,
}

#[derive(Debug, Clone)]
enum TracedArgument {
    Local(usize),
    Input(usize),
}

#[derive(Debug, Clone)]
struct TracedBlock {
    program: Vec<(Primitive, Vec<TracedArgument>)>,
    output: TracedArgument,
    inputs: usize,
}

impl TracedBlock {
    fn new(inputs: usize) -> Self {
        Self {
            program: Vec::new(),
            output: TracedArgument::Input(0),
            inputs,
        }
    }
}

#[derive(Debug, Clone)]
struct ExprTrace {
    block: Arc<RefCell<TracedBlock>>,
}

impl ExprTrace {
    fn new(inputs: usize) -> Self {
        Self {
            block: Arc::new(RefCell::new(TracedBlock::new(inputs))),
        }
    }

    fn add_to_program(&self, prim: Primitive, inputs: Vec<TracedArgument>) -> usize {
        let mut block = self.block.borrow_mut();
        block.program.push((prim, inputs));
        block.program.len() - 1
    }
}

impl Trace for ExprTrace {
    type Value = ExprTracer;

    fn primitive(&self, prim: Primitive, inputs: &[&Self::Value]) -> Self::Value {
        let mut expr_inputs = Vec::new();
        for i in inputs {
            expr_inputs.push(i.value.clone());
        }
        let index = self.add_to_program(prim, expr_inputs);
        ExprTracer {
            trace: self.clone(),
            value: TracedArgument::Local(index),
        }
    }
}

// fn jit<T: Trace + Clone, GF>(fun: GF) -> impl Fn(&T, &T::Value) -> T::Value
// where
//     GF: Fn(&ExprTrace, &<ExprTrace as Trace>::Value) -> <ExprTrace as Trace>::Value,
// {
//     move |_trace, value| {
//         // let value_for_grad = GradTracer::<T::Value>::new(value.clone(), trace.constant(1.0));
//         // let grad_trace = GradTrace {
//         //     inner: trace.clone(),
//         // };
//         // let result = fun(&grad_trace, &value_for_grad);
//         // result.grad
//         value.clone()
//     }
// }

fn test_fn<T: Trace>(trace: &T, value: &T::Value) -> T::Value {
    let x_2 = trace.mul(value, value);
    let x_times_4 = trace.mul(&trace.constant(4.0), value);
    trace.add(&trace.add(&x_2, &x_times_4), &trace.constant(6.0))
}

fn main() {
    let x_for_eval = 3.0;

    let eval_trace = EvalTrace {};
    println!(
        "Result of x^2+4x+6 at 3: {0:?}",
        test_fn(&eval_trace, &x_for_eval)
    );

    let grad_test_fn = grad::<EvalTrace, _>(test_fn);

    println!(
        "Gradient of x^2+4x+6 at 3: {0:?}",
        grad_test_fn(&eval_trace, &x_for_eval)
    );

    let grad2_test_fn = grad::<EvalTrace, _>(grad::<GradTrace<EvalTrace>, _>(test_fn));
    println!(
        "Second gradient of x^2+4x+6 at 3: {0:?}",
        grad2_test_fn(&eval_trace, &x_for_eval)
    );
}

#[test]
fn eval_poly() {
    let x = 3.0;
    assert_eq!(test_fn(&EvalTrace {}, &x), 27.0);
}
#[test]
fn eval_grad_poly() {
    let x = 3.0;
    let grad_test_fn = grad::<EvalTrace, _>(test_fn);
    assert_eq!(grad_test_fn(&EvalTrace {}, &x), 10.0);
}

#[test]
fn eval_grad_grad_poly() {
    let x = 3.0;
    let grad2_test_fn = grad::<EvalTrace, _>(grad::<GradTrace<EvalTrace>, _>(test_fn));
    assert_eq!(grad2_test_fn(&EvalTrace {}, &x), 2.0);
}
