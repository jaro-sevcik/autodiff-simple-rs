use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
enum Primitive {
    Mul,
    Add,
    Constant(f32),
    Block(TracedBlock),
}

// Represents the state captured during tracing.
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

// Direct evaluation trace. This does not have any state because it just computes
// the values directly.
#[derive(Debug, Clone)]
struct EvalTrace {}

// Helper for evaluating a jitted block.
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

// Captured grad trace graph.
#[derive(Debug, Clone)]
struct LinearGraph<T> {
    expressions: Vec<LinearExpression<T>>,
}

impl<T> LinearGraph<T> {
    fn new() -> Self {
        Self {
            expressions: Vec::new(),
        }
    }

    fn constant(value: T) -> LinearExpressionValue<T> {
        LinearExpressionValue::Constant(value)
    }
}

#[derive(Debug, Clone)]
enum LinearExpressionValue<T> {
    Input(usize),
    ExpressionIndex(usize),
    Constant(T),
}

#[derive(Debug, Clone)]
enum LinearExpression<T> {
    Mul(LinearExpressionValue<T>, T),
    Add(LinearExpressionValue<T>, LinearExpressionValue<T>),
}

struct LinearExpressionEvaluationContext<T: Trace> {
    values: Vec<T::Tracer>,
    inputs: Vec<T::Tracer>,
}

impl<T: Trace> LinearExpressionEvaluationContext<T> {
    fn new(inputs: usize, expression_count: usize, trace: &T) -> Self {
        let mut values = vec![trace.constant(0.0); expression_count];
        if !values.is_empty() {
            values[expression_count - 1] = trace.constant(1.0)
        }

        Self {
            values,
            inputs: vec![trace.constant(0.0); inputs],
        }
    }

    fn add_to_value(
        &mut self,
        index: &LinearExpressionValue<T::Tracer>,
        value: &T::Tracer,
        trace: &T,
    ) {
        match index {
            LinearExpressionValue::ExpressionIndex(i) => {
                self.values[*i] = trace.add(&self.values[*i], value)
            }
            LinearExpressionValue::Input(i) => self.inputs[*i] = trace.add(&self.inputs[*i], value),
            LinearExpressionValue::Constant(_) => (),
        }
    }
}

#[derive(Debug, Clone)]
struct GradTracer<T> {
    value: T,
    grad: LinearExpressionValue<T>,
}

impl<T> GradTracer<T> {
    fn new(value: T, grad: LinearExpressionValue<T>) -> Self {
        Self { value, grad }
    }
}

#[derive(Debug, Clone)]
struct GradTrace<Inner: Trace> {
    inner: Inner,
    linear_grad_graph: Rc<RefCell<LinearGraph<Inner::Tracer>>>,
}

impl<Inner: Trace> GradTrace<Inner> {
    fn new(inner: Inner) -> Self {
        let linear_grad_graph = Rc::new(RefCell::new(LinearGraph::new()));
        Self {
            inner,
            linear_grad_graph,
        }
    }

    fn add_expression(&self, op: LinearExpression<Inner::Tracer>) -> usize {
        let mut graph = self.linear_grad_graph.borrow_mut();
        graph.expressions.push(op);
        graph.expressions.len() - 1
    }

    fn linear_add(
        &self,
        lhs: LinearExpressionValue<Inner::Tracer>,
        rhs: LinearExpressionValue<Inner::Tracer>,
    ) -> LinearExpressionValue<Inner::Tracer> {
        let index = self.add_expression(LinearExpression::Add(lhs, rhs));
        LinearExpressionValue::ExpressionIndex(index)
    }

    fn linear_mul(
        &self,
        lhs: LinearExpressionValue<Inner::Tracer>,
        rhs: Inner::Tracer,
    ) -> LinearExpressionValue<Inner::Tracer> {
        let index = self.add_expression(LinearExpression::Mul(lhs, rhs));
        LinearExpressionValue::ExpressionIndex(index)
    }

    fn linear_const(&self, value: Inner::Tracer) -> LinearExpressionValue<Inner::Tracer> {
        LinearGraph::<Inner::Tracer>::constant(value)
    }

    fn evaluate_graph(&self) -> Inner::Tracer {
        let graph = self.linear_grad_graph.borrow();
        let mut context = LinearExpressionEvaluationContext::<Inner>::new(
            1,
            graph.expressions.len(),
            &self.inner,
        );

        let mut position = graph.expressions.len();
        for e in graph.expressions.iter().rev() {
            position -= 1;
            let v = context.values[position].clone();
            match e {
                LinearExpression::Add(lhs, rhs) => {
                    context.add_to_value(lhs, &v, &self.inner);
                    context.add_to_value(rhs, &v, &self.inner);
                }
                LinearExpression::Mul(grad, c) => {
                    context.add_to_value(grad, &self.inner.mul(&v, c), &self.inner)
                }
            }
        }
        context.inputs[0].clone()
    }
}

impl<Inner: Trace> Trace for GradTrace<Inner> {
    type Tracer = GradTracer<Inner::Tracer>;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Self::Tracer {
        match prim {
            Primitive::Constant(c) => {
                assert_eq!(inputs.len(), 0);
                Self::Tracer::new(
                    self.inner.constant(*c),
                    self.linear_const(self.inner.constant(0.0)),
                )
            }
            Primitive::Add => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.add(&inputs[0].value, &inputs[1].value);
                let grad_value = self.linear_add(inputs[0].grad.clone(), inputs[1].grad.clone());
                Self::Tracer::new(value, grad_value)
            }
            Primitive::Mul => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.mul(&inputs[0].value, &inputs[1].value);
                let grad_value = self.linear_add(
                    self.linear_mul(inputs[0].grad.clone(), inputs[1].value.clone()),
                    self.linear_mul(inputs[1].grad.clone(), inputs[0].value.clone()),
                );
                Self::Tracer::new(value, grad_value)
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
        let grad_trace = GradTrace::new(trace.clone());
        let value_for_grad =
            GradTracer::<T::Tracer>::new(value.clone(), LinearExpressionValue::Input(0));
        fun(&grad_trace, &value_for_grad); // TODO: We should perhaps use the result to evaluate.
        grad_trace.evaluate_graph()
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
    let jit_of_grad_test_fn = jit::<EvalTrace, _>(grad::<ExprTrace, _>(test_fn));
    println!(
        "Jit of grad of x^2+4x+6 at 3: {0:?}",
        jit_of_grad_test_fn(&eval_trace, &x_for_eval)
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

#[test]
fn eval_jit_grad_poly() {
    let x = 3.0;
    let jit_of_grad_test_fn = jit::<EvalTrace, _>(grad::<ExprTrace, _>(test_fn));
    assert_eq!(jit_of_grad_test_fn(&EvalTrace {}, &x), 10.0);
}

#[test]
fn eval_jit_poly() {
    let x = 3.0;
    let jit_test_fn = jit::<EvalTrace, _>(test_fn);
    assert_eq!(jit_test_fn(&EvalTrace {}, &x), 27.0);
}
