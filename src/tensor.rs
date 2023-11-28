use std::cmp::PartialEq;
use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::sync::Arc;

#[derive(PartialEq, Debug)]
pub enum DType {
    F32,
    I32,
    Error,
}

#[derive(Debug, Clone, Copy)]
pub struct ShapeDimension {
    pub size: usize,
    pub stride: usize,
}

impl PartialEq for ShapeDimension {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
    }
    fn ne(&self, other: &Self) -> bool {
        self.size != other.size
    }
}

pub enum TensorStorage {
    Float32(Vec<f32>),
    Int32(Vec<i32>),
    Error(String),
}

#[derive(PartialEq, Debug, Clone)]
pub struct Shape(Vec<ShapeDimension>);

impl Shape {
    pub fn dims(&self) -> usize {
        self.0.len()
    }

    pub fn dim(&self, dim: usize) -> ShapeDimension {
        self.0[dim]
    }

    pub fn size(&self) -> usize {
        self.0.iter().fold(1usize, |size, dim| size * dim.size)
    }
}

#[derive(Clone)]
pub struct Tensor {
    storage: Arc<TensorStorage>,
    shape: Shape,
}

impl Tensor {
    pub fn dtype(&self) -> DType {
        match *self.storage {
            TensorStorage::Float32(_) => DType::F32,
            TensorStorage::Int32(_) => DType::I32,
            TensorStorage::Error(_) => DType::I32,
        }
    }

    pub fn error(shape: &Shape, message: &str) -> Self {
        Self {
            shape: shape.clone(),
            storage: Arc::new(TensorStorage::Error(message.to_string())),
        }
    }

    pub fn scalar_f32(value: f32) -> Self {
        Self {
            shape: Shape(Vec::new()),
            storage: Arc::new(TensorStorage::Float32(vec![value])),
        }
    }

    pub fn scalar_i32(value: i32) -> Self {
        Self {
            shape: Shape(Vec::new()),
            storage: Arc::new(TensorStorage::Int32(vec![value])),
        }
    }

    pub fn new_f32(values: &[f32], shape: &[usize]) -> Self {
        let mut size = 1usize;
        let mut shape_dims = Vec::new();
        for i in (0..shape.len()).rev() {
            let dim_size = shape[i];
            shape_dims.push(ShapeDimension {
                size: dim_size,
                stride: size,
            });
            size *= dim_size;
        }
        shape_dims.reverse();
        assert_eq!(values.len(), size);

        let storage = Arc::new(TensorStorage::Float32(values.to_vec()));
        Self {
            storage,
            shape: Shape(shape_dims),
        }
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        let storage = match self.storage.as_ref() {
            TensorStorage::Float32(s) => s,
            _ => panic!("to_vec_f32 needs tensor with f32 storage"),
        };

        let size = self.shape.size();
        let dims = self.shape.dims();
        let mut res = Vec::with_capacity(size);
        let mut index = vec![(0usize, 0usize); self.shape.dims()];
        if size == 1 {
            return vec![storage[0]];
        }
        'outer: loop {
            res.push(storage[index[dims - 1].1]);

            // Now try to advance the index.
            let mut current = dims - 1;
            loop {
                index[current].0 += 1;
                index[current].1 += self.shape.dim(current).stride;
                if index[current].0 < self.shape.dim(current).size {
                    break;
                }
                if current == 0 {
                    break 'outer;
                }
                current -= 1;
            }
            for i in current + 1..dims {
                index[i] = (0usize, index[i - 1].1);
            }
        }
        res
    }

    pub fn get_f32_item(&self, index: &[usize]) -> f32 {
        if let TensorStorage::Float32(storage) = self.storage.as_ref() {
            if self.shape.dims() != index.len() {
                panic!("Invalid index (dimension count mismatch");
            }

            let mut offset = 0;
            for i in 0..index.len() {
                offset += index[i] * self.shape.dim(i).stride;
            }
            storage[offset]
        } else {
            panic!("Cannot extract f32 value from non-float32 tensor");
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.shape.0.iter().map(|s| s.size).collect()
    }

    fn get_error_if_not_compatible(&self, other: &Self) -> Option<Self> {
        // Propagate errors.
        if let TensorStorage::Error(_) = *self.storage {
            return Some(Tensor {
                shape: self.shape.clone(),
                storage: self.storage.clone(),
            });
        }
        if let TensorStorage::Error(_) = *other.storage {
            return Some(Tensor {
                shape: other.shape.clone(),
                storage: other.storage.clone(),
            });
        }
        // Check shapes and types.
        if self.shape != other.shape {
            return Some(Self::error(
                &self.shape,
                "Incompatible tensors - different shapes",
            ));
        }
        if self.dtype() != other.dtype() {
            return Some(Self::error(
                &self.shape,
                "Incompatible tensors - different types",
            ));
        }
        None
    }

    pub fn add(&self, rhs: &Self) -> Self {
        if let Some(e) = self.get_error_if_not_compatible(rhs) {
            return e;
        }
        match (self.storage.as_ref(), rhs.storage.as_ref()) {
            (TensorStorage::Float32(lhs_storage), TensorStorage::Float32(rhs_storage)) => {
                let (shape, result) = pointwise_binop::<f32, _>(
                    &lhs_storage,
                    &self.shape,
                    &rhs_storage,
                    &rhs.shape,
                    f32::add,
                    0.0f32,
                );
                Tensor {
                    shape,
                    storage: Arc::new(TensorStorage::Float32(result)),
                }
            }
            (TensorStorage::Int32(lhs_storage), TensorStorage::Int32(rhs_storage)) => {
                let (shape, result) = pointwise_binop::<i32, _>(
                    &lhs_storage,
                    &self.shape,
                    &rhs_storage,
                    &rhs.shape,
                    i32::add,
                    0,
                );
                Tensor {
                    shape,
                    storage: Arc::new(TensorStorage::Int32(result)),
                }
            }
            (_, _) => Self::error(&self.shape, "Unsupported combination for addition"),
        }
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        if let Some(e) = self.get_error_if_not_compatible(rhs) {
            return e;
        }
        match (self.storage.as_ref(), rhs.storage.as_ref()) {
            (TensorStorage::Float32(lhs_storage), TensorStorage::Float32(rhs_storage)) => {
                let (shape, result) = pointwise_binop::<f32, _>(
                    &lhs_storage,
                    &self.shape,
                    &rhs_storage,
                    &rhs.shape,
                    f32::mul,
                    0.0f32,
                );
                Tensor {
                    shape,
                    storage: Arc::new(TensorStorage::Float32(result)),
                }
            }
            (TensorStorage::Int32(lhs_storage), TensorStorage::Int32(rhs_storage)) => {
                let (shape, result) = pointwise_binop::<i32, _>(
                    &lhs_storage,
                    &self.shape,
                    &rhs_storage,
                    &rhs.shape,
                    i32::mul,
                    0,
                );
                Tensor {
                    shape,
                    storage: Arc::new(TensorStorage::Int32(result)),
                }
            }
            (_, _) => Self::error(&self.shape, "Unsupported combination for addition"),
        }
    }

    pub fn matmul(&self, other: &Self) -> Self {
        // Propagate errors.
        if let TensorStorage::Error(_) = *self.storage {
            return Tensor {
                shape: self.shape.clone(),
                storage: self.storage.clone(),
            };
        }
        if let TensorStorage::Error(_) = *other.storage {
            return Tensor {
                shape: other.shape.clone(),
                storage: other.storage.clone(),
            };
        }
        if self.shape.dims() != other.shape.dims() {
            return Tensor::error(&self.shape, "matmul requires tensor of same dimensions");
        }
        let d = self.shape.dims();
        if d < 2 {
            return Tensor::error(&self.shape, "matmul at least two dimensions");
        }
        for i in 0..d - 2 {
            if self.shape.dim(i).size != other.shape.dim(i).size {
                return Tensor::error(
                    &self.shape,
                    "matmul requires same size of dimensions [0..dims-2]",
                );
            }
        }
        if self.shape.dim(d - 1).size != other.shape.dim(d - 2).size {
            return Tensor::error(
                &self.shape,
                "matmul requires matching lhs[d-1] and rhs[d-2] dimension sizes ",
            );
        }
        match (self.storage.as_ref(), other.storage.as_ref()) {
            (TensorStorage::Float32(lhs_storage), TensorStorage::Float32(rhs_storage)) => {
                let m = self.shape.dim(d - 2).size;
                let n = other.shape.dim(d - 1).size;
                let k = self.shape.dim(d - 1).size;

                // Build the result storage and the result shape.
                let mut result_shape_builder = ContiguousShapeBuilder::new();
                result_shape_builder.add_low(n);
                result_shape_builder.add_low(m);
                for i in 0..d - 2 {
                    let dim_size = self.shape.dim(i).size;
                    result_shape_builder.add_low(dim_size);
                }
                let result_size = result_shape_builder.size();
                let mut result_storage = vec![0.0f32; result_size];
                let result_shape = result_shape_builder.finish();
                let mut result_offset = 0usize;
                let mut lhs_tensor_index = TensorIndex::new(&self.shape, d - 2);
                let mut rhs_tensor_index = TensorIndex::new(&other.shape, d - 2);

                let lhs_stride = self.shape.dim(d - 1).stride;
                let rhs_stride = other.shape.dim(d - 2).stride;

                while result_offset < result_size {
                    let lhs_offset = lhs_tensor_index.next().unwrap();
                    let rhs_offset = rhs_tensor_index.next().unwrap();

                    for i in 0..m {
                        let lhs_index = lhs_offset + i * self.shape.dim(d - 2).stride;
                        for j in 0..n {
                            let rhs_index = rhs_offset + j * other.shape.dim(d - 1).stride;
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum = sum
                                    + lhs_storage[lhs_index + l * lhs_stride]
                                        * rhs_storage[rhs_index + l * rhs_stride];
                            }
                            result_storage[result_offset] = sum;
                            result_offset += 1;
                        }
                    }
                }
                Self {
                    shape: result_shape,
                    storage: Arc::new(TensorStorage::Float32(result_storage)),
                }
            }
            (_, _) => Self::error(
                &self.shape,
                "Only float32 matrix multiplication is supported",
            ),
        }
    }

    // Transposes the last two axes.
    pub fn transpose(&self) -> Self {
        let mut shape = self.shape.clone();
        let d = shape.dims();
        let t = shape.0[d - 1];
        shape.0[d - 1] = shape.0[d - 2];
        shape.0[d - 2] = t;
        Self {
            shape,
            storage: self.storage.clone(),
        }
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<")?;
        match self.storage.as_ref() {
            TensorStorage::Error(_) => write!(f, "error")?,
            TensorStorage::Float32(storage) => write!(f, "{:?}", storage)?,
            TensorStorage::Int32(storage) => write!(f, "{:?}", storage)?,
        }
        write!(f, ", shape: {:?}", self.shape.0)?;
        write!(f, ">")?;
        Ok(())
    }
}

fn pointwise_binop<T: Copy, F>(
    lhs: &[T],
    lhs_shape: &Shape,
    rhs: &[T],
    rhs_shape: &Shape,
    op: F,
    init: T,
) -> (Shape, Vec<T>)
where
    F: Fn(T, T) -> T,
{
    let d = lhs_shape.dims();
    // Fast path for scalars.
    if d == 0 {
        return (lhs_shape.clone(), vec![op(lhs[0], rhs[0])]);
    }

    let ShapeDimension {
        size: batch_size,
        stride: lhs_stride,
    } = lhs_shape.dim(d - 1);
    let rhs_stride = rhs_shape.dim(d - 1).stride;

    let mut result_shape_builder = ContiguousShapeBuilder::new();
    for d in lhs_shape.0.iter().rev() {
        result_shape_builder.add_low(d.size);
    }
    let mut result_storage = vec![init; result_shape_builder.size()];
    let result_shape = result_shape_builder.finish();

    let mut lhs_index = TensorIndex::new(lhs_shape, d - 1);
    let mut rhs_index = TensorIndex::new(rhs_shape, d - 1);
    let mut result_offset = 0usize;

    while let Some(mut lhs_offset) = lhs_index.next() {
        let mut rhs_offset = rhs_index.next().unwrap();
        for _i in 0..batch_size {
            result_storage[result_offset] = op(lhs[lhs_offset], rhs[rhs_offset]);
            lhs_offset += lhs_stride;
            rhs_offset += rhs_stride;
            result_offset += 1;
        }
    }
    (result_shape, result_storage)
}

#[derive(Debug)]
struct TensorIndexElement {
    shape: ShapeDimension,
    i: usize,
    offset: usize,
}

#[derive(Debug)]
struct TensorIndex {
    // |index| stores the state of iteration in reverse dimension order.
    index: Vec<TensorIndexElement>,
    first: bool,
}

impl TensorIndex {
    pub fn new(shape: &Shape, dimensions: usize) -> Self {
        let mut index = Vec::with_capacity(dimensions);
        for dim in shape.0[..dimensions].iter().rev() {
            index.push(TensorIndexElement {
                shape: dim.clone(),
                i: 0,
                offset: 0,
            });
        }
        TensorIndex { index, first: true }
    }

    pub fn next(&mut self) -> Option<usize> {
        if self.first {
            self.first = false;
            return Some(0);
        }
        let dims = self.index.len();
        let mut d = 0;
        loop {
            if d == dims {
                return None;
            }
            let state = &mut self.index[d];
            // Increment the index in the current dimension.
            state.i += 1;
            state.offset += state.shape.stride;
            // If we are still in bounds, we are done incrementing.
            if state.i < state.shape.size {
                break;
            }
            // Otherwise move on to the next dimension.
            d += 1;
        }
        // Now lets reset all the lower dimensions.
        for i in (0..d).rev() {
            self.index[i].i = 0usize;
            self.index[i].offset = self.index[i + 1].offset;
        }

        Some(self.index[0].offset)
    }
}

struct ContiguousShapeBuilder {
    dims: Vec<ShapeDimension>,
    size: usize,
}

impl ContiguousShapeBuilder {
    fn new() -> Self {
        Self {
            dims: Vec::new(),
            size: 1,
        }
    }

    fn add_low(&mut self, dim_size: usize) {
        self.dims.push(ShapeDimension {
            stride: self.size,
            size: dim_size,
        });
        self.size *= dim_size;
    }

    fn size(&self) -> usize {
        self.size
    }

    fn finish(self) -> Shape {
        let mut dims = self.dims;
        dims.reverse();
        Shape(dims)
    }
}

#[test]
fn scalar_add() {
    let t = Tensor::scalar_f32(1.0).add(&Tensor::scalar_f32(2.0));
    assert_eq!(t.get_f32_item(&[]), 3.0);
}

#[test]
fn scalar_mul() {
    let t = Tensor::scalar_f32(2.0).mul(&Tensor::scalar_f32(3.0));
    assert_eq!(t.get_f32_item(&[]), 6.0);
}

#[test]
fn tensor2d_add() {
    let t1 = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = Tensor::new_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let t = t1.add(&t2);
    assert_eq!(&t.to_vec_f32(), &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn tensor2d_add_transposed() {
    let t1 = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).transpose();
    let t2 = Tensor::new_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);

    assert_eq!(&t1.add(&t2).to_vec_f32(), &[11.0, 23.0, 32.0, 44.0]);
    assert_eq!(&t2.add(&t1).to_vec_f32(), &[11.0, 23.0, 32.0, 44.0]);
}

#[test]
fn construct_destruct_0d() {
    let t = Tensor::new_f32(&[1.0], &[]);
    assert_eq!(&t.to_vec_f32(), &[1.0]);
}

#[test]
fn construct_destruct_1d() {
    let t = Tensor::new_f32(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(&t.to_vec_f32(), &[1.0, 2.0, 3.0]);
}

#[test]
fn construct_destruct_2d() {
    let t = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(&t.to_vec_f32(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn construct_destruct_3d() {
    let t = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    assert_eq!(&t.to_vec_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn matmul_2x2_2x2() {
    let t1 = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = Tensor::new_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    assert_eq!(&t1.matmul(&t2).to_vec_f32(), &[70.0, 100.0, 150.0, 220.0]);
}

#[test]
fn matmul_dot_product() {
    let t1 = Tensor::new_f32(&[1.0, 2.0, 3.0], &[1, 3]);
    let t2 = Tensor::new_f32(&[4.0, 3.0, 2.0], &[3, 1]);
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_vec_f32(), &[16.0]);
    assert_eq!(&p.shape(), &[1, 1]);
}

#[test]
fn matmul_2x3_3x1() {
    let t1 = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let t2 = Tensor::new_f32(&[4.0, 3.0, 2.0], &[1, 3]).transpose();
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_vec_f32(), &[16.0, 43.0]);
    assert_eq!(&p.shape(), &[2, 1]);
}

#[test]
fn matmul_2x_1x3_3x1() {
    let t1 = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    let t2 = Tensor::new_f32(&[4.0, 3.0, 2.0, 1.0, 2.0, 3.0], &[2, 1, 3]).transpose();
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_vec_f32(), &[16.0, 32.0]);
    assert_eq!(&p.shape(), &[2, 1, 1]);
}

#[test]
fn matmul_2x_1x2_2x2() {
    let t1 = Tensor::new_f32(&[4.0, 3.0, 2.0, 1.0], &[2, 1, 2]);
    let t2 = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_vec_f32(), &[13.0, 20.0, 17.0, 20.0]);
    assert_eq!(&p.shape(), &[2, 1, 2]);
}

#[test]
fn matmul_2x_1x2_2x2_t() {
    let t1 = Tensor::new_f32(&[4.0, 3.0, 2.0, 1.0], &[2, 1, 2]);
    let t2 = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).transpose();
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_vec_f32(), &[10.0, 24.0, 16.0, 22.0]);
    assert_eq!(&p.shape(), &[2, 1, 2]);
}

#[test]
fn matmul_3x_2x_2x1_1x1_t() {
    let t1 = Tensor::new_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 2, 2, 1],
    );
    let t2 = Tensor::new_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2, 1, 1]);
    let p = t1.matmul(&t2);
    assert_eq!(
        &p.to_vec_f32(),
        &[1.0, 2.0, 6.0, 8.0, 15.0, 18.0, 28.0, 32.0, 45.0, 50.0, 66.0, 72.0]
    );
    assert_eq!(&p.shape(), &[2, 3, 2, 1]);
}
