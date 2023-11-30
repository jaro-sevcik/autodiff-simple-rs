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

    pub fn is_continuous(&self) -> bool {
        let mut last_size = 1usize;
        let mut last_stride = 1usize;
        for ShapeDimension { size, stride } in self.0.iter().rev() {
            if last_size * last_stride != *stride {
                return false;
            }
            last_size = *size;
            last_stride = *stride;
        }
        return true;
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

    pub fn from_scalar_f32(value: f32) -> Self {
        Self {
            shape: Shape(Vec::new()),
            storage: Arc::new(TensorStorage::Float32(vec![value])),
        }
    }

    pub fn from_scalar_i32(value: i32) -> Self {
        Self {
            shape: Shape(Vec::new()),
            storage: Arc::new(TensorStorage::Int32(vec![value])),
        }
    }

    pub fn from_data_f32(values: &[f32], shape: &[usize]) -> Self {
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

    pub fn from_constant_broadcast_f32(constant: f32, shape: &[usize]) -> Self {
        let tensor_shape = shape
            .iter()
            .map(|size| ShapeDimension {
                size: *size,
                stride: 0,
            })
            .collect();
        Self {
            shape: Shape(tensor_shape),
            storage: Arc::new(TensorStorage::Float32(vec![constant])),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::from_constant_broadcast_f32(1.0, shape)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        Self::from_constant_broadcast_f32(0.0, shape)
    }

    pub fn to_data_f32(&self) -> Vec<f32> {
        let storage = match self.storage.as_ref() {
            TensorStorage::Float32(s) => s,
            _ => panic!("to_vec_f32 needs tensor with f32 storage"),
        };

        let dims = self.shape.dims();
        if dims == 0 {
            return vec![storage[0]];
        }
        let mut index = TensorIndex::from_shape(&self.shape, dims - 1);
        let ShapeDimension {
            size: dim_size,
            stride,
        } = self.shape.dim(dims - 1);
        let mut dst_offset = 0;
        let size = self.shape.size();
        let mut data = vec![f32::default(); size];
        while let Some(mut src_offset) = index.next() {
            for _i in 0..dim_size {
                data[dst_offset] = storage[src_offset];
                src_offset += stride;
                dst_offset += 1;
            }
        }
        data
    }

    pub fn get_item_f32(&self, index: &[usize]) -> f32 {
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
                let mut lhs_tensor_index = TensorIndex::from_shape(&self.shape, d - 2);
                let mut rhs_tensor_index = TensorIndex::from_shape(&other.shape, d - 2);

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

    pub fn reshape(&self, shape: &[usize]) -> Self {
        let size = self.shape.size();
        let new_size = shape.iter().fold(1usize, |size, dim| size * dim);
        assert_eq!(size, new_size);
        // If the current shape is continuous, then just reuse the same storage with the new shape.
        let new_shape = ContiguousShapeBuilder::from_sizes(shape).finish();
        if self.shape.is_continuous() {
            return Self {
                shape: new_shape,
                storage: self.storage.clone(),
            };
        }

        // Otherwise we need to copy.
        let dims = self.shape.dims();
        let mut index = TensorIndex::from_shape(&self.shape, dims - 1);
        let ShapeDimension { size, stride } = if dims > 0 {
            self.shape.dim(dims - 1)
        } else {
            ShapeDimension { size: 1, stride: 1 }
        };
        let mut dst_offset = 0;
        match self.storage.as_ref() {
            TensorStorage::Float32(old_storage) => {
                let mut storage = vec![f32::default(); new_size];
                while let Some(mut src_offset) = index.next() {
                    for _ in 0..size {
                        storage[dst_offset] = old_storage[src_offset];
                        src_offset += stride;
                        dst_offset += 1;
                    }
                }
                Self {
                    storage: Arc::new(TensorStorage::Float32(storage)),
                    shape: new_shape,
                }
            }
            TensorStorage::Int32(old_storage) => {
                let mut storage = vec![i32::default(); new_size];
                while let Some(mut src_offset) = index.next() {
                    for _ in 0..size {
                        storage[dst_offset] = old_storage[src_offset];
                        src_offset += stride;
                        dst_offset += 1;
                    }
                }
                Self {
                    storage: Arc::new(TensorStorage::Int32(storage)),
                    shape: new_shape,
                }
            }
            _ => self.clone(),
        }
    }

    pub fn broadcast(&self, dim_size_pairs: &[(usize, usize)]) -> Self {
        let mut shape_dims = self.shape.0.clone();
        let mut last_dim: Option<usize> = None;
        for (dim, size) in dim_size_pairs {
            assert!(
                last_dim.is_none() || last_dim.unwrap() < *dim,
                "Broadcst (dimension, size) pairs must be sorted by dimension index"
            );
            assert_eq!(
                shape_dims[*dim].size, 1usize,
                "Broadcasted dimensions must be of size one, but dimension {} has size {}",
                *dim, shape_dims[*dim].size
            );
            shape_dims[*dim] = ShapeDimension {
                size: *size,
                stride: 0,
            };
            last_dim = Some(*dim)
        }
        Self {
            shape: Shape(shape_dims),
            storage: self.storage.clone(),
        }
    }

    pub fn sum(&self, axis: Option<&[usize]>, keep_dim: bool) -> Self {
        let dims = self.shape.dims();
        let mut other_index = TensorIndex::new();
        let mut sum_index;
        let mut shape_builder = ContiguousShapeBuilder::new();
        if let Some(axis) = axis {
            sum_index = TensorIndex::new();
            let mut finger = axis.len();
            for i in (0..dims).rev() {
                if finger > 0 && axis[finger - 1] == i {
                    finger -= 1;
                    sum_index.add_low(self.shape.dim(i));
                    if keep_dim {
                        shape_builder.add_low(1);
                    }
                } else {
                    other_index.add_low(self.shape.dim(i));
                    shape_builder.add_low(self.shape.dim(i).size);
                };
            }
        } else {
            sum_index = TensorIndex::from_shape(&self.shape, dims);
            if keep_dim {
                for _ in 0..dims {
                    shape_builder.add_low(1);
                }
            }
        }
        let result_size = shape_builder.size();
        match self.storage.as_ref() {
            TensorStorage::Float32(storage) => {
                let mut result_storage = vec![0.0f32; result_size];
                let mut result_offset = 0;
                while let Some(base_offset) = other_index.next() {
                    let mut temp_index = sum_index.clone();
                    let mut sum = 0.0;
                    while let Some(offset) = temp_index.next() {
                        sum += storage[base_offset + offset];
                    }
                    result_storage[result_offset] = sum;
                    result_offset += 1;
                }
                Self {
                    shape: shape_builder.finish(),
                    storage: TensorStorage::Float32(result_storage).into(),
                }
            }
            _ => Self::error(&self.shape, "Sum is only supported for f32 tensors"),
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

    let mut lhs_index = TensorIndex::from_shape(lhs_shape, d - 1);
    let mut rhs_index = TensorIndex::from_shape(rhs_shape, d - 1);
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

#[derive(Debug, Clone)]
struct TensorIndexElement {
    shape: ShapeDimension,
    i: usize,
    offset: usize,
}

#[derive(Debug, Clone)]
struct TensorIndex {
    // |index| stores the state of iteration in reverse dimension order.
    index: Vec<TensorIndexElement>,
    first: bool,
}

impl TensorIndex {
    pub fn from_shape(shape: &Shape, dimensions: usize) -> Self {
        let mut res = Self::with_capacity(dimensions);
        for dim in shape.0[..usize::min(dimensions, shape.0.len())]
            .iter()
            .rev()
        {
            res.add_low(dim.clone());
        }
        res
    }

    pub fn new() -> Self {
        TensorIndex {
            index: Vec::new(),
            first: true,
        }
    }

    pub fn with_capacity(c: usize) -> Self {
        TensorIndex {
            index: Vec::with_capacity(c),
            first: true,
        }
    }

    pub fn add_low(&mut self, dim: ShapeDimension) {
        // Make sure we do not add iteration dimensions after starting iteration.
        assert!(self.first);
        self.index.push(TensorIndexElement {
            shape: dim,
            i: 0,
            offset: 0,
        });
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

    fn from_sizes(sizes: &[usize]) -> Self {
        let mut builder = Self::new();
        for s in sizes {
            builder.add_low(*s);
        }
        builder
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
    let t = Tensor::from_scalar_f32(1.0).add(&Tensor::from_scalar_f32(2.0));
    assert_eq!(t.get_item_f32(&[]), 3.0);
}

#[test]
fn scalar_mul() {
    let t = Tensor::from_scalar_f32(2.0).mul(&Tensor::from_scalar_f32(3.0));
    assert_eq!(t.get_item_f32(&[]), 6.0);
}

#[test]
fn tensor2d_add() {
    let t1 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = Tensor::from_data_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let t = t1.add(&t2);
    assert_eq!(&t.to_data_f32(), &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn tensor2d_add_transposed() {
    let t1 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).transpose();
    let t2 = Tensor::from_data_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);

    assert_eq!(&t1.add(&t2).to_data_f32(), &[11.0, 23.0, 32.0, 44.0]);
    assert_eq!(&t2.add(&t1).to_data_f32(), &[11.0, 23.0, 32.0, 44.0]);
}

#[test]
fn construct_destruct_0d() {
    let t = Tensor::from_data_f32(&[1.0], &[]);
    assert_eq!(&t.to_data_f32(), &[1.0]);
}

#[test]
fn construct_destruct_1d() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(&t.to_data_f32(), &[1.0, 2.0, 3.0]);
}

#[test]
fn construct_destruct_2d() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(&t.to_data_f32(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn construct_destruct_3d() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    assert_eq!(&t.to_data_f32(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn matmul_2x2_2x2() {
    let t1 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = Tensor::from_data_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    assert_eq!(&t1.matmul(&t2).to_data_f32(), &[70.0, 100.0, 150.0, 220.0]);
}

#[test]
fn matmul_dot_product() {
    let t1 = Tensor::from_data_f32(&[1.0, 2.0, 3.0], &[1, 3]);
    let t2 = Tensor::from_data_f32(&[4.0, 3.0, 2.0], &[3, 1]);
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_data_f32(), &[16.0]);
    assert_eq!(&p.shape(), &[1, 1]);
}

#[test]
fn matmul_2x3_3x1() {
    let t1 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let t2 = Tensor::from_data_f32(&[4.0, 3.0, 2.0], &[1, 3]).transpose();
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_data_f32(), &[16.0, 43.0]);
    assert_eq!(&p.shape(), &[2, 1]);
}

#[test]
fn matmul_2x_1x3_3x1() {
    let t1 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    let t2 = Tensor::from_data_f32(&[4.0, 3.0, 2.0, 1.0, 2.0, 3.0], &[2, 1, 3]).transpose();
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_data_f32(), &[16.0, 32.0]);
    assert_eq!(&p.shape(), &[2, 1, 1]);
}

#[test]
fn matmul_2x_1x2_2x2() {
    let t1 = Tensor::from_data_f32(&[4.0, 3.0, 2.0, 1.0], &[2, 1, 2]);
    let t2 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_data_f32(), &[13.0, 20.0, 17.0, 20.0]);
    assert_eq!(&p.shape(), &[2, 1, 2]);
}

#[test]
fn matmul_2x_1x2_2x2_t() {
    let t1 = Tensor::from_data_f32(&[4.0, 3.0, 2.0, 1.0], &[2, 1, 2]);
    let t2 =
        Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).transpose();
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_data_f32(), &[10.0, 24.0, 16.0, 22.0]);
    assert_eq!(&p.shape(), &[2, 1, 2]);
}

#[test]
fn matmul_3x_2x_2x1_1x1_t() {
    let t1 = Tensor::from_data_f32(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 2, 2, 1],
    );
    let t2 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2, 1, 1]);
    let p = t1.matmul(&t2);
    assert_eq!(
        &p.to_data_f32(),
        &[1.0, 2.0, 6.0, 8.0, 15.0, 18.0, 28.0, 32.0, 45.0, 50.0, 66.0, 72.0]
    );
    assert_eq!(&p.shape(), &[2, 3, 2, 1]);
}

#[test]
fn reshape_2x2_to_4() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let r = t.reshape(&[4]);
    assert_eq!(&r.to_data_f32(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&r.shape(), &[4]);
}

#[test]
fn reshape_1x1_to_scalar() {
    let t = Tensor::from_data_f32(&[1.0], &[1, 1]);
    let r = t.reshape(&[]);
    assert_eq!(&r.to_data_f32(), &[1.0]);
    assert_eq!(&r.shape(), &[]);
}

#[test]
fn reshape_3x2_t_to_6() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).transpose();
    let r = t.reshape(&[6]);
    assert_eq!(&r.to_data_f32(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    assert_eq!(&r.shape(), &[6]);
}

#[test]
fn ones_3x2() {
    let t = Tensor::ones(&[3, 2]);
    assert_eq!(&t.to_data_f32(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&t.shape(), &[3, 2]);
}

#[test]
fn sum_3x2_to_2() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = t.sum(Some(&[1]), false);
    assert_eq!(&s.to_data_f32(), &[6.0, 15.0]);
    assert_eq!(&s.shape(), &[2]);
}

#[test]
fn sum_3x2_to_3() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = t.sum(Some(&[0]), false);
    assert_eq!(&s.to_data_f32(), &[5.0, 7.0, 9.0]);
    assert_eq!(&s.shape(), &[3]);
}

#[test]
fn sum_3x2_to_2_keepdim() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = t.sum(Some(&[1]), true);
    assert_eq!(&s.to_data_f32(), &[6.0, 15.0]);
    assert_eq!(&s.shape(), &[2, 1]);
}

#[test]
fn sum_2x2x2_to_2_keepdim() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let s = t.sum(Some(&[1]), true);
    assert_eq!(&s.to_data_f32(), &[4.0, 6.0, 12.0, 14.0]);
    assert_eq!(&s.shape(), &[2, 1, 2]);
}

#[test]
fn sum_3x2_t_to_3() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).transpose();
    let s = t.sum(Some(&[1]), false);
    assert_eq!(&s.to_data_f32(), &[5.0, 7.0, 9.0]);
    assert_eq!(&s.shape(), &[3]);
}

#[test]
fn sum_2x2_to_scalar() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let s = t.sum(None, false);
    assert_eq!(&s.to_data_f32(), &[10.0]);
    assert_eq!(&s.shape(), &[]);
}

#[test]
fn sum_2x2_to_scalar_keepdim() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let s = t.sum(None, true);
    assert_eq!(&s.to_data_f32(), &[10.0]);
    assert_eq!(&s.shape(), &[1, 1]);
}

#[test]
fn broadcast_1d() {
    let t = Tensor::from_data_f32(&[1.0], &[1]);
    let r = t.broadcast(&[(0, 4)]);
    assert_eq!(&r.to_data_f32(), &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&r.shape(), &[4]);
}

#[test]
fn broadcast_1x1_to_3x2() {
    let t = Tensor::from_data_f32(&[1.0], &[1, 1]);
    let r = t.broadcast(&[(0, 2), (1, 3)]);
    assert_eq!(&r.to_data_f32(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&r.shape(), &[2, 3]);
}
#[test]

fn broadcast_2x1x2_to_2x2x2() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let r = t.broadcast(&[(1, 2)]);
    assert_eq!(&r.to_data_f32(), &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    assert_eq!(&r.shape(), &[2, 2, 2]);
}
