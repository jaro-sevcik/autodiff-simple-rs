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
struct GradTracer {
    value: f32,
    grad: f32,
}

impl GradTracer {
    fn new(value: f32, grad: f32) -> Self {
        Self{value, grad}
    }
}

impl Tracer for GradTracer {
    fn add(&self, rhs: &Self) -> Self {
        Self::new(self.value + rhs.value, self.grad + rhs.grad)
    }

    fn mul(&self, rhs: &Self) -> Self {
        Self::new(self.value * rhs.value, self.value * rhs.grad + self.grad * rhs.value)
    }
}

struct GradTrace {}

impl Trace for GradTrace {
    type Value = GradTracer;

    fn constant(&self, value: f32) -> Self::Value {
        Self::Value{value, grad: 0.0}
    }
}

fn test_fn<T: Trace>(trace: &T, value: &T::Value) -> T::Value {
    let x_2 = value.mul(value);
    let x_times_4 = trace.constant(4.0).mul(value);
    x_2.add(&x_times_4).add(&trace.constant(6.0))
}

fn main() {
    let x_for_eval = EvalTracer::new(3.0);

    let eval_trace = EvalTrace{};
    println!("Result of x^2+4x+6 at 3: {0:?}", test_fn(&eval_trace, &x_for_eval));

    let grad_trace = GradTrace{};
    let x_for_grad = GradTracer::new(3.0, 1.0);
    println!("Result (value, grad) of x^2+4x+6 at 3: {0:?}", test_fn(&grad_trace, &x_for_grad));
}
