// Represents a traced value. Wrapper for a float.
trait Tracer {
    fn add(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
}

// Represents a trace.
// Contains factory methods for creating tracers.
trait Trace {
    type Value: Tracer;
    fn constant(&self, value: f32) -> Self::Value;
}

#[derive(Debug)]
struct EvalTracer {
    value: f32,
}

impl EvalTracer {
    fn new(value: f32) -> Self {
        Self { value }
    }
}

impl Tracer for EvalTracer {
    fn add(&self, rhs: &Self) -> Self {
        Self::new(self.value + rhs.value)
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self::new(self.value * rhs.value)
    }
}

struct EvalTrace {}

impl Trace for EvalTrace {
    type Value = EvalTracer;

    fn constant(&self, value: f32) -> Self::Value {
        Self::Value::new(value)
    }
}

#[derive(Debug)]
struct GradTracer<T: Tracer> {
    value: T,
    grad: T,
}

impl<T: Tracer> GradTracer<T> {
    fn new(value: T, grad: T) -> Self {
        Self { value, grad }
    }
}

impl<T: Tracer> Tracer for GradTracer<T> {
    fn add(&self, rhs: &Self) -> Self {
        let value = self.value.add(&rhs.value);
        let grad = self.grad.add(&rhs.grad);
        Self::new(value, grad)
    }

    fn mul(&self, rhs: &Self) -> Self {
        let value = self.value.mul(&rhs.value);
        let grad = self.value.mul(&rhs.grad).add(&self.grad.mul(&rhs.value));
        Self::new(value, grad)
    }
}

struct GradTrace<Inner: Trace> {
    inner: Inner,
}

impl<Inner: Trace> Trace for GradTrace<Inner> {
    type Value = GradTracer<Inner::Value>;

    fn constant(&self, value: f32) -> Self::Value {
        Self::Value {
            value: self.inner.constant(value),
            grad: self.inner.constant(0.0),
        }
    }
}

fn test_fn<T: Trace>(trace: &T, value: &T::Value) -> T::Value {
    let x_2 = value.mul(value);
    let x_times_4 = trace.constant(4.0).mul(value);
    x_2.add(&x_times_4).add(&trace.constant(6.0))
}

fn main() {
    let x_for_eval = EvalTracer::new(3.0);

    let eval_trace = EvalTrace {};
    println!(
        "Result of x^2+4x+6 at 3: {0:?}",
        test_fn(&eval_trace, &x_for_eval)
    );

    let x_for_grad = GradTracer::new(x_for_eval, eval_trace.constant(1.0));
    let grad_trace = GradTrace { inner: eval_trace };
    println!(
        "Result (value, grad) of x^2+4x+6 at 3: {0:?}",
        test_fn(&grad_trace, &x_for_grad)
    );

    let x_for_grad2 = GradTracer::new(x_for_grad, grad_trace.constant(1.0));
    let grad2_trace = GradTrace { inner: grad_trace };
    println!(
        "Result (value, grad^2) of x^2+4x+6 at 3: {0:?}",
        test_fn(&grad2_trace, &x_for_grad2)
    );
}
