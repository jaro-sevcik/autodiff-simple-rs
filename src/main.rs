use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
enum Primitive {
    Mul,
    Add,
    Constant(f32),
    Block(TracedBlock),
}

// Represents a trace.
trait Trace {
    type Tracer: Clone;
    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Self::Tracer;

    fn add(&self, lhs: &Self::Tracer, rhs: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::Add, &[lhs, rhs])
    }
    fn mul(&self, lhs: &Self::Tracer, rhs: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::Mul, &[lhs, rhs])
    }
    fn constant(&self, value: f32) -> Self::Tracer {
        self.primitive(&Primitive::Constant(value), &[])
    }
}

#[derive(Debug, Clone)]
struct EvalTrace {}

fn evaluate_block<T: Trace>(trace: &T, b: &TracedBlock, inputs: &[&T::Tracer]) -> T::Tracer {
    assert_eq!(b.inputs, inputs.len());
    let mut locals = Vec::<T::Tracer>::new();
    for (prim, args) in &b.program {
        // Resolve the arguments.
        let mut arg_values = Vec::new();
        for a in args {
            let arg = match a {
                TracedArgument::Input(i) => inputs[*i],
                TracedArgument::Local(i) => &locals[*i],
            };
            arg_values.push(arg)
        }

        // Evaluate the primitive.
        locals.push(trace.primitive(prim, &arg_values));
    }
    // Extract the output.
    match b.output {
        TracedArgument::Input(i) => inputs[i].clone(),
        TracedArgument::Local(i) => locals[i].clone(),
    }
}

impl Trace for EvalTrace {
    type Tracer = f32;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Self::Tracer {
        match prim {
            Primitive::Constant(c) => *c,
            Primitive::Add => inputs[0] + inputs[1],
            Primitive::Mul => inputs[0] * inputs[1],
            Primitive::Block(b) => {
                println!("Evaluating block {:?}", b);
                evaluate_block(self, b, inputs)
            }
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
    type Tracer = GradTracer<Inner::Tracer>;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Self::Tracer {
        match prim {
            Primitive::Constant(c) => {
                assert_eq!(inputs.len(), 0);
                Self::Tracer {
                    value: self.inner.constant(*c),
                    grad: self.inner.constant(0.0),
                }
            }
            Primitive::Add => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.add(&inputs[0].value, &inputs[1].value);
                let grad = self.inner.add(&inputs[0].grad, &inputs[1].grad);
                Self::Tracer::new(value, grad)
            }
            Primitive::Mul => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.mul(&inputs[0].value, &inputs[1].value);
                let grad = self.inner.add(
                    &self.inner.mul(&inputs[0].value, &inputs[1].grad),
                    &self.inner.mul(&inputs[0].grad, &inputs[1].value),
                );
                Self::Tracer::new(value, grad)
            }
            Primitive::Block(b) => evaluate_block(self, b, inputs),
        }
    }
}

fn grad<T: Trace + Clone, GF>(fun: GF) -> impl Fn(&T, &T::Tracer) -> T::Tracer
where
    GF: Fn(&GradTrace<T>, &<GradTrace<T> as Trace>::Tracer) -> <GradTrace<T> as Trace>::Tracer,
{
    move |trace, value| {
        let grad_trace = GradTrace {
            inner: trace.clone(),
        };
        let value_for_grad = GradTracer::<T::Tracer>::new(value.clone(), trace.constant(1.0));
        let result = fun(&grad_trace, &value_for_grad);
        result.grad
    }
}

#[derive(Debug, Clone)]
struct ExprTracer {
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
    block: Rc<RefCell<TracedBlock>>,
}

impl ExprTrace {
    fn new(inputs: usize) -> Self {
        Self {
            block: Rc::new(RefCell::new(TracedBlock::new(inputs))),
        }
    }

    fn add_to_program(&self, prim: Primitive, inputs: Vec<TracedArgument>) -> usize {
        let mut block = self.block.borrow_mut();
        block.program.push((prim, inputs));
        block.program.len() - 1
    }
}

impl Trace for ExprTrace {
    type Tracer = ExprTracer;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Self::Tracer {
        let mut expr_inputs = Vec::new();
        for i in inputs {
            expr_inputs.push(i.value.clone());
        }
        let index = self.add_to_program(prim.clone(), expr_inputs);
        ExprTracer {
            value: TracedArgument::Local(index),
        }
    }
}

fn jit<T: Trace + Clone, GF>(fun: GF) -> impl Fn(&T, &T::Tracer) -> T::Tracer
where
    GF: Fn(&ExprTrace, &<ExprTrace as Trace>::Tracer) -> <ExprTrace as Trace>::Tracer,
{
    move |in_trace, value| {
        let expr_trace = ExprTrace::new(1);
        let param_tracer = ExprTracer {
            value: TracedArgument::Input(0),
        };
        let output = fun(&expr_trace, &param_tracer).value;

        let primitive = Primitive::Block(TracedBlock {
            inputs: 1,
            program: expr_trace.block.borrow().program.clone(),
            output,
        });

        in_trace.primitive(&primitive, &[value])
    }
}

fn test_fn<T: Trace>(trace: &T, value: &T::Tracer) -> T::Tracer {
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

    let jit_test_fn = jit::<EvalTrace, _>(test_fn);
    println!(
        "Jit of x^2+4x+6 at 3: {0:?}",
        jit_test_fn(&eval_trace, &x_for_eval)
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
