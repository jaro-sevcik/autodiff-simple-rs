use log::trace;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

use crate::tensor::{Shape, Tensor};
use crate::trace::*;

// Captured grad trace graph.
#[derive(Clone)]
struct LinearGraph<T> {
    expressions: Vec<(Shape, LinearExpression<T>)>,
    input_shapes: Vec<Shape>,
}

impl<T> LinearGraph<T> {
    fn new(input_shapes: Vec<Shape>) -> Self {
        Self {
            expressions: Vec::new(),
            input_shapes,
        }
    }

    fn shape_for(&self, value: &LinearExpressionValue) -> Shape {
        match value {
            LinearExpressionValue::ExpressionIndex(i) => self.expressions[*i].0.clone(),
            LinearExpressionValue::Input(i) => self.input_shapes[*i].clone(),
            LinearExpressionValue::Zero(shape) => shape.clone(),
        }
    }
}

impl<T> Debug for LinearGraph<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let new_line = if f.alternate() { "\n" } else { "; " };
        for i in 0..self.expressions.len() {
            let e = &self.expressions[i];
            write!(f, "{}%{} <- {:?}", new_line, i, e)?;
        }
        Ok(())
    }
}

#[derive(Clone)]
enum LinearExpressionValue {
    Input(usize),
    ExpressionIndex(usize),
    Zero(Shape),
}

impl Debug for LinearExpressionValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearExpressionValue::Zero(shape) => write!(f, "zero@{:?}", shape),
            LinearExpressionValue::Input(i) => write!(f, "I{}", i),
            LinearExpressionValue::ExpressionIndex(i) => write!(f, "%{}", i),
        }
    }
}

#[derive(Debug, Clone)]
enum LinearExpression<T> {
    Mul(LinearExpressionValue, T),
    MatMulLeft(LinearExpressionValue, T),
    MatMulRight(T, LinearExpressionValue),
    Add(LinearExpressionValue, LinearExpressionValue),
    Reshape(LinearExpressionValue),
}

struct LinearExpressionEvaluationContext<T: Trace> {
    values: Vec<T::Tracer>,
    inputs: Vec<T::Tracer>,
}

impl<T: Trace> LinearExpressionEvaluationContext<T> {
    fn new(input_shapes: &[Shape], expression_shapes: &[Shape], trace: &T) -> Self {
        // TODO create the state with correct shapes!
        // let inputs = input_shapes.iter().map(|s| Tensor::zeros(&s)).collect();
        let values: Vec<<T as Trace>::Tracer> = expression_shapes
            .iter()
            .map(|s| trace.constant(Tensor::zeros(s.clone())))
            .collect();
        let inputs: Vec<<T as Trace>::Tracer> = input_shapes
            .iter()
            .map(|s| trace.constant(Tensor::zeros(s.clone())))
            .collect();
        Self { values, inputs }
    }

    fn add_to_value(&mut self, index: &LinearExpressionValue, value: &T::Tracer, trace: &T) {
        match index {
            LinearExpressionValue::ExpressionIndex(i) => {
                self.values[*i] = trace.add(&self.values[*i], value)
            }
            LinearExpressionValue::Input(i) => self.inputs[*i] = trace.add(&self.inputs[*i], value),
            LinearExpressionValue::Zero(_shape) => (),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GradTracer<T> {
    value: T,
    grad: LinearExpressionValue,
}

impl<T: Shaped> Shaped for GradTracer<T> {
    fn shape(&self) -> Shape {
        self.value.shape()
    }
}

impl<T> GradTracer<T> {
    fn new(value: T, grad: LinearExpressionValue) -> Self {
        Self { value, grad }
    }
}

#[derive(Debug, Clone)]
pub struct GradTrace<Inner: Trace> {
    inner: Inner,
    linear_grad_graph: Rc<RefCell<LinearGraph<Inner::Tracer>>>,
}

impl<Inner: Trace> GradTrace<Inner> {
    fn new(inner: Inner, input_shapes: Vec<Shape>) -> Self {
        let linear_grad_graph = Rc::new(RefCell::new(LinearGraph::new(input_shapes)));
        Self {
            inner,
            linear_grad_graph,
        }
    }

    fn add_expression(&self, shape: Shape, op: LinearExpression<Inner::Tracer>) -> usize {
        let mut graph = self.linear_grad_graph.borrow_mut();
        graph.expressions.push((shape, op));
        graph.expressions.len() - 1
    }

    fn linear_add(
        &self,
        lhs: LinearExpressionValue,
        rhs: LinearExpressionValue,
        shape: Shape,
    ) -> LinearExpressionValue {
        if let LinearExpressionValue::Zero(_) = lhs {
            return rhs;
        }
        if let LinearExpressionValue::Zero(_) = rhs {
            return lhs;
        }
        let index = self.add_expression(shape, LinearExpression::Add(lhs, rhs));
        LinearExpressionValue::ExpressionIndex(index)
    }

    fn linear_matmul_left(
        &self,
        lhs: LinearExpressionValue,
        rhs: Inner::Tracer,
        shape: Shape,
    ) -> LinearExpressionValue {
        if let LinearExpressionValue::Zero(_) = lhs {
            return LinearExpressionValue::Zero(shape);
        }
        let index = self.add_expression(shape, LinearExpression::MatMulLeft(lhs, rhs));
        LinearExpressionValue::ExpressionIndex(index)
    }

    fn linear_matmul_right(
        &self,
        lhs: Inner::Tracer,
        rhs: LinearExpressionValue,
        shape: Shape,
    ) -> LinearExpressionValue {
        if let LinearExpressionValue::Zero(_) = rhs {
            return LinearExpressionValue::Zero(shape);
        }
        let index = self.add_expression(shape, LinearExpression::MatMulRight(lhs, rhs));
        LinearExpressionValue::ExpressionIndex(index)
    }

    fn linear_mul(
        &self,
        lhs: LinearExpressionValue,
        rhs: Inner::Tracer,
        shape: Shape,
    ) -> LinearExpressionValue {
        if let LinearExpressionValue::Zero(_) = lhs {
            return lhs;
        }
        let index = self.add_expression(shape, LinearExpression::Mul(lhs, rhs));
        LinearExpressionValue::ExpressionIndex(index)
    }

    fn linear_reshape(&self, shape: Shape, input: LinearExpressionValue) -> LinearExpressionValue {
        if let LinearExpressionValue::Zero(_) = input {
            return LinearExpressionValue::Zero(shape);
        }
        let index = self.add_expression(shape, LinearExpression::Reshape(input));
        LinearExpressionValue::ExpressionIndex(index)
    }

    fn linear_zero(&self, shape: Shape) -> LinearExpressionValue {
        LinearExpressionValue::Zero(shape)
    }

    fn evaluate_graph(&self, result: LinearExpressionValue) -> Vec<Inner::Tracer> {
        let graph = self.linear_grad_graph.borrow();
        let expression_shapes: Vec<Shape> = graph
            .expressions
            .iter()
            .map(|(shape, _)| shape.clone())
            .collect();
        let mut context = LinearExpressionEvaluationContext::<Inner>::new(
            &graph.input_shapes,
            &expression_shapes,
            &self.inner,
        );

        context.add_to_value(
            &result,
            &self.inner.constant(Tensor::from_scalar_f32(1.0)),
            &self.inner,
        );

        let mut position = graph.expressions.len();
        for (_shape, e) in graph.expressions.iter().rev() {
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
                LinearExpression::MatMulLeft(grad, c) => {
                    context.add_to_value(grad, &self.inner.matmul(&v, c), &self.inner)
                }
                LinearExpression::MatMulRight(c, grad) => {
                    context.add_to_value(grad, &self.inner.matmul(c, &v), &self.inner)
                }
                LinearExpression::Reshape(grad) => {
                    let shape = graph.shape_for(grad);
                    context.add_to_value(grad, &self.inner.reshape(shape, &v), &self.inner)
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
                    self.inner.constant(c.clone()),
                    self.linear_zero(c.shape()),
                )]
            }
            Primitive::Add => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.add(&inputs[0].value, &inputs[1].value);
                let grad_value = self.linear_add(
                    inputs[0].grad.clone(),
                    inputs[1].grad.clone(),
                    value.shape(),
                );
                vec![Self::Tracer::new(value, grad_value)]
            }
            Primitive::Mul => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.mul(&inputs[0].value, &inputs[1].value);
                let shape = value.shape();
                let grad_value = self.linear_add(
                    self.linear_mul(
                        inputs[0].grad.clone(),
                        inputs[1].value.clone(),
                        shape.clone(),
                    ),
                    self.linear_mul(
                        inputs[1].grad.clone(),
                        inputs[0].value.clone(),
                        shape.clone(),
                    ),
                    shape,
                );
                vec![Self::Tracer::new(value, grad_value)]
            }
            Primitive::MatMul => {
                assert_eq!(inputs.len(), 2);
                let value = self.inner.matmul(&inputs[0].value, &inputs[1].value);
                let shape = value.shape();
                let grad_value = self.linear_add(
                    self.linear_matmul_left(
                        inputs[0].grad.clone(),
                        inputs[1].value.clone(),
                        shape.clone(),
                    ),
                    self.linear_matmul_right(
                        inputs[0].value.clone(),
                        inputs[1].grad.clone(),
                        shape.clone(),
                    ),
                    shape,
                );
                vec![Self::Tracer::new(value, grad_value)]
            }
            Primitive::Reshape(shape) => {
                assert_eq!(inputs.len(), 1);
                let value = self.inner.reshape(shape.clone(), &inputs[0].value);
                let grad_value = self.linear_reshape(shape.clone(), inputs[0].grad.clone()); // TODO provide correct operator!
                vec![Self::Tracer::new(value, grad_value)]
            }
            Primitive::Block(b) => evaluate_block(self, b, inputs),
        }
    }
}

pub fn grad<T: Trace + Clone, GF>(
    fun: GF,
    grad_input_count: usize,
) -> impl Fn(&T, &[T::Tracer]) -> Vec<T::Tracer>
where
    GF: Fn(
        &GradTrace<T>,
        &[<GradTrace<T> as Trace>::Tracer],
    ) -> Vec<<GradTrace<T> as Trace>::Tracer>,
    T::Tracer: std::fmt::Debug,
{
    move |trace, values| {
        assert!(grad_input_count <= values.len());
        let grad_input_shapes = values[..grad_input_count]
            .iter()
            .map(|v| v.shape())
            .collect();
        let grad_trace = GradTrace::new(trace.clone(), grad_input_shapes);
        let parameter_tracers: Vec<GradTracer<T::Tracer>> = (0..values.len())
            .map(|i| {
                let grad = if i < grad_input_count {
                    LinearExpressionValue::Input(i)
                } else {
                    LinearExpressionValue::Zero(values[i].shape())
                };
                GradTracer::<T::Tracer>::new(values[i].clone(), grad)
            })
            .collect();
        let result = fun(&grad_trace, &parameter_tracers);
        trace!(
            "Grad linear graph: {:#?}",
            grad_trace.linear_grad_graph.borrow()
        );
        println!(
            "Grad linear graph: {:#?}",
            grad_trace.linear_grad_graph.borrow()
        );
        assert_eq!(result.len(), 1);
        grad_trace.evaluate_graph(result[0].grad.clone())
    }
}

// TODO plug into grad for `sum`.
fn compute_reduce_shape(shape: &[usize], axis: Option<&[usize]>) -> Vec<usize> {
    let dims = shape.len();
    let mut result = Vec::new();
    if let Some(axis) = axis {
        let mut finger = axis.len();
        for i in (0..dims).rev() {
            if finger > 0 && axis[finger - 1] == i {
                finger -= 1;
                result.push(1);
            } else {
                result.push(shape[i]);
            };
        }
    } else {
        for _ in 0..dims {
            result.push(1);
        }
    }
    result
}
