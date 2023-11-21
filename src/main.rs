use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone)]
enum Primitive {
    Mul,
    Add,
    Constant(f32),
    Block(TracedBlock),
    Projection(usize),
}

// Represents the state captured during tracing.
trait Trace {
    type Tracer: Clone;
    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Vec<Self::Tracer>;

    fn add(&self, lhs: &Self::Tracer, rhs: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::Add, &[lhs, rhs])[0].clone()
    }
    fn mul(&self, lhs: &Self::Tracer, rhs: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::Mul, &[lhs, rhs])[0].clone()
    }
    fn constant(&self, value: f32) -> Self::Tracer {
        self.primitive(&Primitive::Constant(value), &[])[0].clone()
    }
}

// Direct evaluation trace. This does not have any state because it just computes
// the values directly.
#[derive(Debug, Clone)]
struct EvalTrace {}

// Helper for evaluating a jitted block.
fn evaluate_block<T: Trace>(trace: &T, b: &TracedBlock, inputs: &[&T::Tracer]) -> Vec<T::Tracer> {
    assert_eq!(b.inputs, inputs.len());
    let mut locals = Vec::<Vec<T::Tracer>>::new();
    for (prim, args) in &b.program {
        // Resolve the arguments.
        let mut arg_values = Vec::new();

        // Handle the projection specially.
        if let Primitive::Projection(i) = prim {
            assert_eq!(args.len(), 1);
            if let TracedBlockVar::Local(local) = args[0] {
                return vec![locals[local][*i].clone()];
            } else {
                panic!("Invalid projection from function input.")
            }
        }

        for a in args {
            let arg = match a {
                TracedBlockVar::Input(i) => inputs[*i],
                TracedBlockVar::Local(i) => &locals[*i][0],
            };
            arg_values.push(arg)
        }

        // Evaluate the primitive.
        locals.push(trace.primitive(prim, &arg_values));
    }
    // Extract the output.
    b.outputs
        .iter()
        .map(|o| match o {
            TracedBlockVar::Input(i) => inputs[*i].clone(),
            TracedBlockVar::Local(i) => locals[*i][0].clone(),
        })
        .collect()
}

impl Trace for EvalTrace {
    type Tracer = f32;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Vec<Self::Tracer> {
        match prim {
            Primitive::Constant(c) => vec![*c],
            Primitive::Add => vec![inputs[0] + inputs[1]],
            Primitive::Mul => vec![inputs[0] * inputs[1]],
            Primitive::Block(b) => evaluate_block(self, b, inputs),
            Primitive::Projection(i) => vec![inputs[*i].clone()],
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

    fn evaluate_graph(&self, input_count: usize) -> Vec<Inner::Tracer> {
        let graph = self.linear_grad_graph.borrow();
        let mut context = LinearExpressionEvaluationContext::<Inner>::new(
            input_count,
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
        context.inputs
    }
}

impl<Inner: Trace> Trace for GradTrace<Inner> {
    type Tracer = GradTracer<Inner::Tracer>;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Vec<Self::Tracer> {
        match prim {
            Primitive::Constant(c) => {
                assert_eq!(inputs.len(), 0);
                vec![Self::Tracer::new(
                    self.inner.constant(*c),
                    self.linear_const(self.inner.constant(0.0)),
                )]
            }
            Primitive::Add => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.add(&inputs[0].value, &inputs[1].value);
                let grad_value = self.linear_add(inputs[0].grad.clone(), inputs[1].grad.clone());
                vec![Self::Tracer::new(value, grad_value)]
            }
            Primitive::Mul => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.mul(&inputs[0].value, &inputs[1].value);
                let grad_value = self.linear_add(
                    self.linear_mul(inputs[0].grad.clone(), inputs[1].value.clone()),
                    self.linear_mul(inputs[1].grad.clone(), inputs[0].value.clone()),
                );
                vec![Self::Tracer::new(value, grad_value)]
            }
            Primitive::Block(b) => evaluate_block(self, b, inputs),
            Primitive::Projection(i) => vec![inputs[*i].clone()],
        }
    }
}

fn grad<T: Trace + Clone, GF>(fun: GF) -> impl Fn(&T, &[T::Tracer]) -> Vec<T::Tracer>
where
    GF: Fn(
        &GradTrace<T>,
        &[<GradTrace<T> as Trace>::Tracer],
    ) -> Vec<<GradTrace<T> as Trace>::Tracer>,
{
    move |trace, values| {
        let grad_trace = GradTrace::new(trace.clone());
        let mut parameter_tracers = Vec::new();
        for v in values {
            let tracer = GradTracer::<T::Tracer>::new(
                v.clone(),
                LinearExpressionValue::Input(parameter_tracers.len()),
            );
            parameter_tracers.push(tracer);
        }
        let result = fun(&grad_trace, &parameter_tracers);
        // TODO: We should perhaps use the result to evaluate.
        assert_eq!(result.len(), 1);
        grad_trace.evaluate_graph(values.len())
    }
}

#[derive(Debug, Clone)]
struct ExprTracer {
    variable: TracedBlockVar,
}

#[derive(Debug, Clone)]
enum TracedBlockVar {
    Local(usize),
    Input(usize),
}

#[derive(Debug, Clone)]
struct TracedBlock {
    program: Vec<(Primitive, Vec<TracedBlockVar>)>,
    outputs: Vec<TracedBlockVar>,
    inputs: usize,
}

impl TracedBlock {
    fn new(inputs: usize) -> Self {
        Self {
            program: Vec::new(),
            outputs: Vec::new(),
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

    fn add_to_program(&self, prim: Primitive, inputs: Vec<TracedBlockVar>) -> usize {
        let mut block = self.block.borrow_mut();
        block.program.push((prim, inputs));
        block.program.len() - 1
    }
}

impl Trace for ExprTrace {
    type Tracer = ExprTracer;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Vec<Self::Tracer> {
        let mut expr_inputs = Vec::new();
        for i in inputs {
            expr_inputs.push(i.variable.clone());
        }
        let index = self.add_to_program(prim.clone(), expr_inputs);
        vec![ExprTracer {
            variable: TracedBlockVar::Local(index),
        }]
    }
}

fn jit<T: Trace + Clone, GF>(fun: GF) -> impl Fn(&T, &[T::Tracer]) -> Vec<T::Tracer>
where
    GF: Fn(&ExprTrace, &[<ExprTrace as Trace>::Tracer]) -> Vec<<ExprTrace as Trace>::Tracer>,
{
    move |in_trace, values| {
        let expr_trace = ExprTrace::new(1);
        // Prepare the arguments as expression tracers.
        let parameter_tracers: Vec<ExprTracer> = (0..values.len())
            .map(|i| ExprTracer {
                variable: TracedBlockVar::Input(i),
            })
            .collect();
        // Compute the expression with the expression trace.
        let result: Vec<ExprTracer> = fun(&expr_trace, &parameter_tracers);
        // Extract the outputs.
        let outputs = result.iter().map(|r| r.variable.clone()).collect();

        // Pass the compiled expression to the underlying trace
        // as a "TracedBlock" primitive.
        let primitive = Primitive::Block(TracedBlock {
            inputs: values.len(),
            program: expr_trace.block.borrow().program.clone(),
            outputs,
        });
        let value_refs: Vec<&T::Tracer> = values.iter().collect();
        in_trace.primitive(&primitive, &value_refs)
    }
}

fn test_fn<T: Trace>(trace: &T, values: &[T::Tracer]) -> Vec<T::Tracer> {
    let value = &values[0];
    let x_2 = trace.mul(value, value);
    let x_times_4 = trace.mul(&trace.constant(4.0), value);
    vec![trace.add(&trace.add(&x_2, &x_times_4), &trace.constant(6.0))]
}

fn main() {
    let x_for_eval = 3.0;

    let eval_trace = EvalTrace {};
    println!(
        "Result of x^2+4x+6 at 3: {0:?}",
        test_fn(&eval_trace, &[x_for_eval])
    );

    let grad_test_fn = grad::<EvalTrace, _>(test_fn);

    println!(
        "Gradient of x^2+4x+6 at 3: {0:?}",
        grad_test_fn(&eval_trace, &[x_for_eval])
    );

    let grad2_test_fn = grad::<EvalTrace, _>(grad::<GradTrace<EvalTrace>, _>(test_fn));
    println!(
        "Second gradient of x^2+4x+6 at 3: {0:?}",
        grad2_test_fn(&eval_trace, &[x_for_eval])
    );

    let jit_test_fn = jit::<EvalTrace, _>(test_fn);
    println!(
        "Jit of x^2+4x+6 at 3: {0:?}",
        jit_test_fn(&eval_trace, &[x_for_eval])
    );
    let jit_of_grad_test_fn = jit::<EvalTrace, _>(grad::<ExprTrace, _>(test_fn));
    println!(
        "Jit of grad of x^2+4x+6 at 3: {0:?}",
        jit_of_grad_test_fn(&eval_trace, &[x_for_eval])
    );
}

#[cfg(test)]
fn poly_fn<T: Trace>(trace: &T, values: &[T::Tracer]) -> Vec<T::Tracer> {
    let value = &values[0];
    let x_2 = trace.mul(value, value);
    let x_times_2 = trace.mul(&trace.constant(2.0), value);
    vec![trace.add(&trace.add(&x_2, &x_times_2), &trace.constant(-3.0))]
}

#[test]
fn eval_poly() {
    let x = 3.0;
    assert_eq!(poly_fn(&EvalTrace {}, &[x]), [12.0]);
}

#[test]
fn eval_grad_poly() {
    let x = 3.0;
    let grad_poly_fn = grad::<EvalTrace, _>(poly_fn);
    assert_eq!(grad_poly_fn(&EvalTrace {}, &[x]), [8.0]);
}

#[test]
fn eval_grad_grad_poly() {
    let x = 3.0;
    let grad2_poly_fn = grad::<EvalTrace, _>(grad::<GradTrace<EvalTrace>, _>(poly_fn));
    assert_eq!(grad2_poly_fn(&EvalTrace {}, &[x]), [2.0]);
}

#[test]
fn eval_jit_grad_poly() {
    let x = 3.0;
    let jit_of_grad_poly_fn = jit::<EvalTrace, _>(grad::<ExprTrace, _>(poly_fn));
    assert_eq!(jit_of_grad_poly_fn(&EvalTrace {}, &[x]), [8.0]);
}

#[test]
fn eval_jit_poly() {
    let x = 3.0;
    let jit_poly_fn = jit::<EvalTrace, _>(poly_fn);
    assert_eq!(jit_poly_fn(&EvalTrace {}, &[x]), [12.0]);
}

#[cfg(test)]
fn test_multivar_mul_fn<T: Trace>(trace: &T, values: &[T::Tracer]) -> Vec<T::Tracer> {
    vec![trace.mul(&values[0], &values[1])]
}

#[test]
fn eval_grad_multivar_mul() {
    let x = 3.0;
    let y = 4.0;
    let grad_test_fn = grad::<EvalTrace, _>(test_multivar_mul_fn);
    assert_eq!(grad_test_fn(&EvalTrace {}, &[x, y]), [4.0, 3.0]);
}

#[test]
fn eval_jit_grad_multivar_mul() {
    let x = 3.0;
    let y = 4.0;
    let jit_of_grad_test_fn = jit::<EvalTrace, _>(grad::<ExprTrace, _>(test_multivar_mul_fn));
    assert_eq!(jit_of_grad_test_fn(&EvalTrace {}, &[x, y]), [4.0, 3.0]);
}
