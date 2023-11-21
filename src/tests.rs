use crate::eval_trace::EvalTrace;
use crate::expr_trace::{jit, ExprTrace};
use crate::grad::{grad, GradTrace};
use crate::trace::Trace;

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
fn eval_jit_poly() {
    let x = 3.0;
    let jit_poly_fn = jit::<EvalTrace, _>(poly_fn);
    assert_eq!(jit_poly_fn(&EvalTrace {}, &[x]), [12.0]);
}

#[test]
fn eval_jit_grad_poly() {
    let x = 3.0;
    let jit_of_grad_poly_fn = jit::<EvalTrace, _>(grad::<ExprTrace, _>(poly_fn));
    assert_eq!(jit_of_grad_poly_fn(&EvalTrace {}, &[x]), [8.0]);
}

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
    let x = 3.0;
    assert_eq!(test_embed_jit_fn(&EvalTrace {}, &[x]), [36.0]);
}

#[test]
fn composed_jit_grad_multivar_mul() {
    let x = 3.0;
    let jit_composed_fn = jit::<EvalTrace, _>(test_embed_jit_fn);
    assert_eq!(jit_composed_fn(&EvalTrace {}, &[x]), [36.0]);
}
