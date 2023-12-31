use log::trace;

use std::cell::RefCell;
use std::rc::Rc;

use crate::tensor::Shape;
use crate::trace::*;

#[derive(Debug, Clone)]
pub struct ExprTracer {
    variable: TracedBlockVar,
    shape: Shape,
}

impl Shaped for ExprTracer {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }
}

#[derive(Debug, Clone)]
pub struct ExprTrace {
    block: Rc<RefCell<TracedBlock>>,
}

impl ExprTrace {
    fn new(input_shapes: Vec<Shape>) -> Self {
        Self {
            block: Rc::new(RefCell::new(TracedBlock::new(input_shapes))),
        }
    }

    fn add_to_program(&self, prim: Primitive, inputs: Vec<TracedBlockVar>) -> usize {
        let mut block = self.block.borrow_mut();
        block.program.push((prim, inputs));
        block.program.len() - 1
    }
}

impl Trace for ExprTrace {
    type Tracer = ExprTracer;

    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Vec<Self::Tracer> {
        let mut expr_inputs = Vec::new();
        for i in inputs {
            expr_inputs.push(i.variable.clone());
        }
        let index = self.add_to_program(prim.clone(), expr_inputs);
        let output_shapes = prim.output_shapes(inputs);
        (0..output_shapes.len())
            .map(|output| ExprTracer {
                variable: TracedBlockVar::Local(index, output),
                shape: output_shapes[output].clone(),
            })
            .collect()
    }
}

pub fn jit<T: Trace + Clone, GF>(fun: GF) -> impl Fn(&T, &[T::Tracer]) -> Vec<T::Tracer>
where
    GF: Fn(&ExprTrace, &[<ExprTrace as Trace>::Tracer]) -> Vec<<ExprTrace as Trace>::Tracer>,
{
    move |in_trace, values| {
        let input_shapes: Vec<Shape> = values.iter().map(|v| v.shape()).collect();
        let expr_trace = ExprTrace::new(input_shapes.clone());
        // Prepare the arguments as expression tracers.
        let parameter_tracers: Vec<ExprTracer> = (0..values.len())
            .map(|i| ExprTracer {
                variable: TracedBlockVar::Input(i),
                shape: values[i].shape(),
            })
            .collect();
        // Compute the expression with the expression trace.
        let result: Vec<ExprTracer> = fun(&expr_trace, &parameter_tracers);
        // Extract the outputs.
        let outputs = result
            .iter()
            .map(|r| (r.shape(), r.variable.clone()))
            .collect();

        // Pass the compiled expression to the underlying trace
        // as a "TracedBlock" primitive.
        let block = TracedBlock {
            input_shapes,
            program: expr_trace.block.borrow().program.clone(),
            outputs,
        };
        trace!("Jitted block:\n{:#?}", block);
        let value_refs: Vec<&T::Tracer> = values.iter().collect();
        in_trace.primitive(&Primitive::Block(block), &value_refs)
    }
}
