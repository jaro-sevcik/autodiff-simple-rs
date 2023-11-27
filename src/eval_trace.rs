use crate::tensor::Tensor;
use crate::trace::*;

// Direct evaluation trace. This does not have any state because it just computes
// the values directly.
#[derive(Debug, Clone)]
pub struct EvalTrace {}

impl Trace for EvalTrace {
    type Tracer = Tensor;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Vec<Self::Tracer> {
        match prim {
            Primitive::Constant(c) => vec![*c],
            Primitive::Add => vec![inputs[0].add(inputs[1])],
            Primitive::Mul => vec![inputs[0].mul(inputs[1])],
            Primitive::Block(b) => evaluate_block(self, b, inputs),
        }
    }
}
