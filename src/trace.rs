#[derive(Debug, Clone)]
pub enum Primitive {
    Mul,
    Add,
    Constant(f32),
    Block(TracedBlock),
}

impl Primitive {
    pub fn output_count(&self) -> usize {
        match self {
            Primitive::Block(b) => b.outputs.len(),
            _ => 1,
        }
    }
}

// Represents the state captured during tracing.
pub trait Trace {
    type Tracer: Clone;
    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Vec<Self::Tracer>;

    fn add(&self, lhs: &Self::Tracer, rhs: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::Add, &[lhs, rhs])[0].clone()
    }
    fn mul(&self, lhs: &Self::Tracer, rhs: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::Mul, &[lhs, rhs])[0].clone()
    }
    fn constant(&self, value: f32) -> Self::Tracer {
        self.primitive(&Primitive::Constant(value), &[])[0].clone()
    }
}

#[derive(Debug, Clone)]
pub enum TracedBlockVar {
    // Local is identified by the [equation-index, output-index] pair.
    Local(usize, usize),
    Input(usize),
}

#[derive(Debug, Clone)]
pub struct TracedBlock {
    pub program: Vec<(Primitive, Vec<TracedBlockVar>)>,
    pub outputs: Vec<TracedBlockVar>,
    pub input_count: usize,
}

impl TracedBlock {
    pub fn new(input_count: usize) -> Self {
        Self {
            program: Vec::new(),
            outputs: Vec::new(),
            input_count,
        }
    }
}

// Helper for evaluating a jitted block.
pub fn evaluate_block<T: Trace>(
    trace: &T,
    b: &TracedBlock,
    inputs: &[&T::Tracer],
) -> Vec<T::Tracer> {
    assert_eq!(b.input_count, inputs.len());
    let mut locals = Vec::<Vec<T::Tracer>>::new();
    for (prim, args) in &b.program {
        // Resolve the arguments.
        let mut arg_values = Vec::new();

        for a in args {
            let arg = match a {
                TracedBlockVar::Input(i) => inputs[*i],
                TracedBlockVar::Local(equation, output) => &locals[*equation][*output],
            };
            arg_values.push(arg)
        }

        // Evaluate the primitive.
        locals.push(trace.primitive(prim, &arg_values));
    }
    // Extract the output.
    b.outputs
        .iter()
        .map(|o| match o {
            TracedBlockVar::Input(i) => inputs[*i].clone(),
            TracedBlockVar::Local(equation, output) => locals[*equation][*output].clone(),
        })
        .collect()
}
