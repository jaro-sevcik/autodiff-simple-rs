mod eval_trace;
mod expr_trace;
mod grad;
mod trace;

#[cfg(test)]
mod tests;

use eval_trace::*;
use expr_trace::*;
use grad::*;
use trace::*;

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
