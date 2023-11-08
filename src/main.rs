// Represents a traced value. Wrapper for a float.
trait Tracer {
    fn add(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;
}

// Represents a trace.
// Contains factory methods for creating tracers.
trait Trace {
    type Value: Tracer + Clone;
    fn constant(&self, value: f32) -> Self::Value;
}

impl Tracer for f32 {
    fn add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    fn mul(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

#[derive(Debug, Clone)]
struct EvalTrace {}

impl Trace for EvalTrace {
    type Value = f32;

    fn constant(&self, value: f32) -> Self::Value {
        value
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
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

fn grad<T: Trace + Clone, GF>(fun: GF) -> impl Fn(&T, &T::Value) -> T::Value
where
    GF: Fn(&GradTrace<T>, &<GradTrace<T> as Trace>::Value) -> <GradTrace<T> as Trace>::Value,
{
    move |trace, value| {
        let value_for_grad = GradTracer::<T::Value>::new(value.clone(), trace.constant(1.0));
        let grad_trace = GradTrace {
            inner: trace.clone(),
        };
        let result = fun(&grad_trace, &value_for_grad);
        result.grad
    }
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