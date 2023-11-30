use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub enum Primitive {
    Mul,
    Add,
    MatMul,
    Reshape(Vec<usize>),
    Constant(Tensor),
    Block(TracedBlock),
}

impl Primitive {
    pub fn output_count(&self) -> usize {
        match self {
            Primitive::Block(b) => b.outputs.len(),
            _ => 1,
        }
    }

    pub fn output_shapes<T: Shaped>(&self, input_shapes: &[&T]) -> Vec<Vec<usize>> {
        match self {
            Primitive::Mul => vec![input_shapes[0].shape()],
            Primitive::Add => vec![input_shapes[0].shape()],
            Primitive::MatMul => {
                let mut result_shape = input_shapes[0].shape();
                let result_len = result_shape.len();
                let rhs_shape = input_shapes[1].shape();
                result_shape[result_len - 1] = rhs_shape[rhs_shape.len()];
                vec![result_shape]
            }
            Primitive::Reshape(shape) => vec![shape.clone()],
            Primitive::Constant(t) => vec![t.shape()],
            Primitive::Block(b) => b.outputs.iter().map(|(shape, _)| shape.clone()).collect(),
        }
    }
}

pub trait Shaped {
    fn shape(&self) -> Vec<usize>;
}

// Represents the state captured during tracing.
pub trait Trace {
    type Tracer: Clone + Shaped;
    fn primitive(&self, prim: &Primitive, inputs: &[&Self::Tracer]) -> Vec<Self::Tracer>;

    fn add(&self, lhs: &Self::Tracer, rhs: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::Add, &[lhs, rhs])[0].clone()
    }
    fn mul(&self, lhs: &Self::Tracer, rhs: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::Mul, &[lhs, rhs])[0].clone()
    }
    fn constant(&self, value: Tensor) -> Self::Tracer {
        self.primitive(&Primitive::Constant(value), &[])[0].clone()
    }
    fn matmul(&self, lhs: &Self::Tracer, rhs: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::MatMul, &[lhs, rhs])[0].clone()
    }
    fn reshape(&self, shape: &[usize], input: &Self::Tracer) -> Self::Tracer {
        self.primitive(&Primitive::Reshape(shape.to_vec()), &[input])[0].clone()
    }
}

#[derive(Debug, Clone)]
pub enum TracedBlockVar {
    // Local is identified by the [equation-index, output-index] pair.
    Local(usize, usize),
    Input(usize),
}

impl TracedBlockVar {
    fn to_string(&self) -> String {
        match self {
            Self::Input(i) => format!("I{}", i),
            Self::Local(i, j) => {
                if *j == 0 {
                    format!("%{}", i)
                } else {
                    format!("%{}#{}", i, j)
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct TracedBlock {
    pub program: Vec<(Primitive, Vec<TracedBlockVar>)>,
    // Output is a pair of a shape and a location.
    pub outputs: Vec<(Vec<usize>, TracedBlockVar)>,
    pub input_shapes: Vec<Vec<usize>>,
}

impl TracedBlock {
    pub fn new(input_shapes: Vec<Vec<usize>>) -> Self {
        Self {
            program: Vec::new(),
            outputs: Vec::new(),
            input_shapes,
        }
    }
}

impl std::fmt::Debug for TracedBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_list();
        d.entry(&ExprInputCountFormatHelper(&self.input_shapes));
        d.entry(&ExprOutputsFormatHelper(&self.outputs));
        for i in 0..self.program.len() {
            let (prim, inputs) = &self.program[i];
            d.entry(&EquationFormatHelper(i, &prim, &inputs));
        }
        d.finish()?;
        Ok(())
    }
}

// Helper for evaluating a jitted block.
pub fn evaluate_block<T: Trace>(
    trace: &T,
    b: &TracedBlock,
    inputs: &[&T::Tracer],
) -> Vec<T::Tracer> {
    // TODO check the shapes, too!
    assert_eq!(b.input_shapes.len(), inputs.len());
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
        .map(|(_shape, output)| match output {
            TracedBlockVar::Input(i) => inputs[*i].clone(),
            TracedBlockVar::Local(equation, output) => locals[*equation][*output].clone(),
        })
        .collect()
}

struct EquationFormatHelper<'a>(usize, &'a Primitive, &'a Vec<TracedBlockVar>);
impl<'a> std::fmt::Debug for EquationFormatHelper<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_list = self
            .2
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<String>>()
            .join(" ");
        match &self.1 {
            Primitive::Block(b) => {
                if f.alternate() {
                    write!(f, "%{} <- Block {:#?} {}", self.0, b, input_list)?
                } else {
                    write!(f, "%{} <- Block {:?} {}", self.0, b, input_list)?
                }
            }
            _ => write!(f, "%{} <- {:?} {}", self.0, self.1, input_list)?,
        }
        Ok(())
    }
}

struct ExprInputCountFormatHelper<'a>(&'a [Vec<usize>]);
impl<'a> std::fmt::Debug for ExprInputCountFormatHelper<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Input shapes: {:?}", self.0)
    }
}

struct ExprOutputsFormatHelper<'a>(&'a Vec<(Vec<usize>, TracedBlockVar)>);
impl<'a> std::fmt::Debug for ExprOutputsFormatHelper<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let output_list = self
            .0
            .iter()
            .map(|(shape, i)| format!("{}@{:?}", i.to_string(), shape))
            .collect::<Vec<String>>()
            .join(" ");
        write!(f, "Outputs: {}", output_list)
    }
}
