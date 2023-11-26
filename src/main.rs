mod error;
mod eval_trace;
mod expr_trace;
mod grad;
mod tensor;
mod trace;

#[cfg(test)]
mod tests;

use log::trace;

use eval_trace::EvalTrace;
use expr_trace::{jit, ExprTrace};
use grad::{grad, GradTrace};
use trace::Trace;

fn test_fn<T: Trace>(trace: &T, values: &[T::Tracer]) -> Vec<T::Tracer> {
    let value = &values[0];
    let x_2 = trace.mul(value, value);
    let x_times_4 = trace.mul(&trace.constant(4.0), value);
    vec![trace.add(&trace.add(&x_2, &x_times_4), &trace.constant(6.0))]
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

fn main() {
    env_logger::builder().format_timestamp(None).init();
    trace!("Started");

    let x_for_eval = 3.0;

    let eval_trace = EvalTrace {};
    println!(
        "Result of x^2+4x+6 at 3: {0:?}",
        test_fn(&eval_trace, &[x_for_eval])
    );

    let grad_test_fn = grad::<EvalTrace, _>(test_fn, 1);

    println!(
        "Gradient of x^2+4x+6 at 3: {0:?}",
        grad_test_fn(&eval_trace, &[x_for_eval])
    );

    let grad2_test_fn = grad::<EvalTrace, _>(grad::<GradTrace<EvalTrace>, _>(test_fn, 1), 1);
    println!(
        "Second gradient of x^2+4x+6 at 3: {0:?}",
        grad2_test_fn(&eval_trace, &[x_for_eval])
    );

    let jit_test_fn = jit::<EvalTrace, _>(test_fn);
    println!(
        "Jit of x^2+4x+6 at 3: {0:?}",
        jit_test_fn(&eval_trace, &[x_for_eval])
    );
    let jit_of_grad_test_fn = jit::<EvalTrace, _>(grad::<ExprTrace, _>(test_fn, 1));
    println!(
        "Jit of grad of x^2+4x+6 at 3: {0:?}",
        jit_of_grad_test_fn(&eval_trace, &[x_for_eval])
    );

    let jit_composed_fn = jit::<EvalTrace, _>(test_embed_jit_fn);
    assert_eq!(jit_composed_fn(&EvalTrace {}, &[x_for_eval]), [36.0]);
}
