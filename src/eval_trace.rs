use crate::trace::*;

// Direct evaluation trace. This does not have any state because it just computes
// the values directly.
#[derive(Debug, Clone)]
pub struct EvalTrace {}

impl Trace for EvalTrace {
    type Tracer = f32;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Vec<Self::Tracer> {
        match prim {
            Primitive::Constant(c) => vec![*c],
            Primitive::Add => vec![inputs[0] + inputs[1]],
            Primitive::Mul => vec![inputs[0] * inputs[1]],
            Primitive::Block(b) => evaluate_block(self, b, inputs),
        }
    }
}