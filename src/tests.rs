use crate::eval_trace::EvalTrace;
use crate::expr_trace::{jit, ExprTrace};
use crate::grad::{grad, GradTrace};
use crate::tensor::{Shape, Tensor};
use crate::trace::{Shaped, Trace};

#[cfg(test)]
#[ctor::ctor]
fn init_logger() {
    env_logger::builder().format_timestamp(None).init();
}

fn poly_fn<T: Trace>(trace: &T, values: &[T::Tracer]) -> Vec<T::Tracer> {
    let value = &values[0];
    let x_2 = trace.mul(value, value);
    let x_times_2 = trace.mul(&trace.constant(Tensor::from_scalar_f32(2.0)), value);
    vec![trace.add(
        &trace.add(&x_2, &x_times_2),
        &trace.constant(Tensor::from_scalar_f32(-3.0)),
    )]
}

fn to_scalars(t: &[Tensor]) -> Vec<f32> {
    t.iter().map(|t| t.get_item_f32(&[])).collect()
}

#[test]
fn eval_poly() {
    let x = Tensor::from_scalar_f32(3.0);
    assert_eq!(to_scalars(&poly_fn(&EvalTrace {}, &[x])), [12.0]);
}

#[test]
fn eval_grad_poly() {
    let x = Tensor::from_scalar_f32(3.0);
    let grad_poly_fn = grad::<EvalTrace, _>(poly_fn, 1);
    assert_eq!(to_scalars(&grad_poly_fn(&EvalTrace {}, &[x])), [8.0]);
}

#[test]
fn eval_grad_grad_poly() {
    let x = Tensor::from_scalar_f32(3.0);
    let grad2_poly_fn = grad::<EvalTrace, _>(grad::<GradTrace<EvalTrace>, _>(poly_fn, 1), 1);
    assert_eq!(to_scalars(&grad2_poly_fn(&EvalTrace {}, &[x])), [2.0]);
}

#[test]
fn eval_jit_poly() {
    let x = Tensor::from_scalar_f32(3.0);
    let jit_poly_fn = jit::<EvalTrace, _>(poly_fn);
    assert_eq!(to_scalars(&jit_poly_fn(&EvalTrace {}, &[x])), [12.0]);
}

#[test]
fn eval_jit_grad_poly() {
    let x = Tensor::from_scalar_f32(3.0);
    let jit_of_grad_poly_fn = jit::<EvalTrace, _>(grad::<ExprTrace, _>(poly_fn, 1));
    assert_eq!(to_scalars(&jit_of_grad_poly_fn(&EvalTrace {}, &[x])), [8.0]);
}

fn test_multivar_mul_fn<T: Trace>(trace: &T, values: &[T::Tracer]) -> Vec<T::Tracer> {
    vec![trace.mul(&values[0], &values[1])]
}

#[test]
fn eval_grad_multivar_mul() {
    let x = Tensor::from_scalar_f32(3.0);
    let y = Tensor::from_scalar_f32(4.0);
    let grad_test_fn = grad::<EvalTrace, _>(test_multivar_mul_fn, 2);
    assert_eq!(
        to_scalars(&grad_test_fn(&EvalTrace {}, &[x, y])),
        [4.0, 3.0]
    );
}

#[test]
fn eval_partial_grad_multivar_mul() {
    let x = Tensor::from_scalar_f32(3.0);
    let y = Tensor::from_scalar_f32(4.0);
    let grad_test_fn = grad::<EvalTrace, _>(test_multivar_mul_fn, 1);
    assert_eq!(to_scalars(&grad_test_fn(&EvalTrace {}, &[x, y])), [4.0]);
}

#[test]
fn eval_jit_grad_multivar_mul() {
    let x = Tensor::from_scalar_f32(3.0);
    let y = Tensor::from_scalar_f32(4.0);
    let jit_of_grad_test_fn = jit::<EvalTrace, _>(grad::<ExprTrace, _>(test_multivar_mul_fn, 2));
    assert_eq!(
        to_scalars(&jit_of_grad_test_fn(&EvalTrace {}, &[x, y])),
        [4.0, 3.0]
    );
}

fn test_multi_output_fn<T: Trace>(trace: &T, values: &[T::Tracer]) -> Vec<T::Tracer> {
    let x2 = trace.mul(&values[0], &values[0]);
    let x3 = trace.mul(&x2, &values[0]);
    vec![x2, x3]
}

fn test_embed_jit_fn<T: Trace + Clone>(trace: &T, values: &[T::Tracer]) -> Vec<T::Tracer> {
    let jitted_multi_out = jit::<T, _>(test_multi_output_fn);

    let res = jitted_multi_out(trace, &[values[0].clone()]);
    let sum = trace.add(&res[0], &res[1]);

    vec![sum]
}

#[test]
fn eval_with_jit_grad_multivar_mul() {
    let x = Tensor::from_scalar_f32(3.0);
    assert_eq!(to_scalars(&test_embed_jit_fn(&EvalTrace {}, &[x])), [36.0]);
}

#[test]
fn composed_jit_grad_multivar_mul() {
    let x = Tensor::from_scalar_f32(3.0);
    let jit_composed_fn = jit::<EvalTrace, _>(test_embed_jit_fn);
    assert_eq!(to_scalars(&jit_composed_fn(&EvalTrace {}, &[x])), [36.0]);
}

fn weighted_sum_fn<T: Trace>(trace: &T, inputs: &[T::Tracer]) -> Vec<T::Tracer> {
    let values = &inputs[0];
    let weights = &inputs[1];
    let size = values.shape().size();
    let value_vector = trace.reshape(Shape::from([1, size]), values);
    let weight_vector = trace.reshape(Shape::from([size, 1]), weights);
    let product = trace.matmul(&value_vector, &weight_vector);
    let res = trace.reshape(Shape::from_iter([]), &product);
    vec![res]
}

#[test]
fn eval_trivial_sum_via_matmul() {
    let x = Tensor::from_data_f32(&[1.0, 2.0], &[2]);
    let w = Tensor::from_data_f32(&[1.0, 1.0], &[2]);
    assert_eq!(to_scalars(&weighted_sum_fn(&EvalTrace {}, &[x, w])), [3.0]);
}

#[test]
fn eval_grad_trivial_sum_via_matmul() {
    let x = Tensor::from_data_f32(&[1.0, 2.0], &[2]);
    let w = Tensor::from_data_f32(&[1.0, 1.0], &[2]);
    let grad_trivial_sum_via_matmul_fn = grad::<EvalTrace, _>(weighted_sum_fn, 1);
    let result = grad_trivial_sum_via_matmul_fn(&EvalTrace {}, &[x, w]);
    assert_eq!(result[0].to_data_f32(), [1.0, 1.0]);
    assert_eq!(result[0].shape().as_ref(), &[2]);
}

#[test]
fn eval_jit_trivial_sum_via_matmul() {
    let x = Tensor::from_data_f32(&[1.0, 2.0], &[2]);
    let w = Tensor::from_data_f32(&[1.0, 1.0], &[2]);
    let jit_sum_via_matmul_fn = jit::<EvalTrace, _>(weighted_sum_fn);
    assert_eq!(
        to_scalars(&jit_sum_via_matmul_fn(&EvalTrace {}, &[x, w])),
        [3.0]
    );
}

#[test]
fn eval_jit_grad_trivial_sum_via_matmul() {
    let x = Tensor::from_data_f32(&[1.0, 2.0], &[2]);
    let w = Tensor::from_data_f32(&[1.0, 1.0], &[2]);
    let jit_of_grad_sum_via_matmul_fn =
        jit::<EvalTrace, _>(grad::<ExprTrace, _>(weighted_sum_fn, 1));
    let result = jit_of_grad_sum_via_matmul_fn(&EvalTrace {}, &[x, w]);
    assert_eq!(result[0].to_data_f32(), [1.0, 1.0]);
    assert_eq!(result[0].shape().as_ref(), &[2]);
}
