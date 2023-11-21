use std::cell::RefCell;
use std::rc::Rc;

use crate::trace::*;

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
pub struct GradTracer<T> {
    value: T,
    grad: LinearExpressionValue<T>,
}

impl<T> GradTracer<T> {
    fn new(value: T, grad: LinearExpressionValue<T>) -> Self {
        Self { value, grad }
    }
}

#[derive(Debug, Clone)]
pub struct GradTrace<Inner: Trace> {
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
        }
    }
}

pub fn grad<T: Trace + Clone, GF>(fun: GF) -> impl Fn(&T, &[T::Tracer]) -> Vec<T::Tracer>
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
