use std::cmp::PartialEq;
use std::convert::From;
use std::fmt::Debug;
use std::iter::{zip, Iterator};
use std::ops::{Add, Index, Mul};
use std::sync::Arc;

#[derive(PartialEq, Debug)]
pub enum DType {
    F32,
    I32,
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
}

pub enum TensorStorage {
    Float32(Vec<f32>),
    Int32(Vec<i32>),
}

#[derive(PartialEq, Debug, Clone)]
pub struct ShapeStride {
    shape: Shape,
    strides: Vec<usize>,
}

impl From<Vec<ShapeDimension>> for ShapeStride {
    fn from(shape_dims: Vec<ShapeDimension>) -> Self {
        let mut sizes = Vec::with_capacity(shape_dims.len());
        let mut strides = Vec::with_capacity(shape_dims.len());
        for s in shape_dims {
            sizes.push(s.size);
            strides.push(s.stride);
        }
        Self {
            shape: Shape::from(sizes),
            strides,
        }
    }
}

impl ShapeStride {
    pub fn new(shape: Shape, strides: Vec<usize>) -> Self {
        assert_eq!(shape.dims(), strides.len());
        ShapeStride { shape, strides }
    }

    pub fn dims(&self) -> usize {
        self.shape.dims()
    }

    pub fn dim(&self, dim: usize) -> ShapeDimension {
        ShapeDimension {
            size: self.shape[dim],
            stride: self.strides[dim],
        }
    }

    pub fn stride(&self, dim: usize) -> usize {
        self.strides[dim]
    }

    pub fn size(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    pub fn total_size(&self) -> usize {
        self.shape.size()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut last_size = 1usize;
        let mut last_stride = 1usize;
        for i in (0..self.shape.dims()).rev() {
            let size = self.shape[i];
            let stride = self.strides[i];
            if last_size * last_stride != stride {
                return false;
            }
            last_size = size;
            last_stride = stride;
        }
        true
    }

    pub fn to_shape_dimensions(&self) -> Vec<ShapeDimension> {
        zip(self.shape.as_ref(), &self.strides)
            .map(|(size, stride)| ShapeDimension {
                size: *size,
                stride: *stride,
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Shape(Arc<Vec<usize>>);

impl Shape {
    pub fn dims(&self) -> usize {
        self.0.len()
    }

    pub fn transpose(&self) -> Shape {
        let l = self.dims();
        assert!(self.dims() >= 2);
        let mut result = self.0.as_ref().clone();
        result.swap(l - 1, l - 2);
        Self::from(result)
    }

    pub fn size(&self) -> usize {
        self.0.iter().fold(1, usize::mul)
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(value: [usize; N]) -> Self {
        Self(Vec::from(value).into())
    }
}

impl PartialEq<[usize]> for Shape {
    fn eq(&self, other: &[usize]) -> bool {
        self.as_ref() == other
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Self(Arc::new(v))
    }
}

impl FromIterator<usize> for Shape {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        Self(iter.into_iter().collect::<Vec<usize>>().into())
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl Index<usize> for Shape {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[derive(Clone)]
pub struct Tensor {
    storage: Arc<TensorStorage>,
    shape_stride: ShapeStride,
}

impl Tensor {
    pub fn dtype(&self) -> DType {
        match *self.storage {
            TensorStorage::Float32(_) => DType::F32,
            TensorStorage::Int32(_) => DType::I32,
        }
    }

    pub fn from_scalar_f32(value: f32) -> Self {
        Self {
            shape_stride: ContiguousShapeBuilder::new().finish(),
            storage: Arc::new(TensorStorage::Float32(vec![value])),
        }
    }

    pub fn from_scalar_i32(value: i32) -> Self {
        Self {
            shape_stride: ContiguousShapeBuilder::new().finish(),
            storage: Arc::new(TensorStorage::Int32(vec![value])),
        }
    }

    pub fn from_data_f32(values: &[f32], shape: &[usize]) -> Self {
        let storage = Arc::new(TensorStorage::Float32(values.to_vec()));
        Self {
            storage,
            shape_stride: ContiguousShapeBuilder::from_sizes(shape).finish(),
        }
    }

    pub fn from_constant_broadcast_f32(constant: f32, shape: Shape) -> Self {
        let strides = vec![0usize; shape.dims()];
        Self {
            shape_stride: ShapeStride::new(shape, strides),
            storage: Arc::new(TensorStorage::Float32(vec![constant])),
        }
    }

    pub fn ones(shape: Shape) -> Self {
        Self::from_constant_broadcast_f32(1.0, shape)
    }

    pub fn zeros(shape: Shape) -> Self {
        Self::from_constant_broadcast_f32(0.0, shape)
    }

    pub fn to_data_f32(&self) -> Vec<f32> {
        let storage = match self.storage.as_ref() {
            TensorStorage::Float32(s) => s,
            _ => panic!("to_vec_f32 needs tensor with f32 storage"),
        };

        let dims = self.shape_stride.dims();
        if dims == 0 {
            return vec![storage[0]];
        }
        let mut index = TensorIndex::from_shape(&self.shape_stride, dims - 1);
        let ShapeDimension {
            size: dim_size,
            stride,
        } = self.shape_stride.dim(dims - 1);
        let mut dst_offset = 0;
        let size = self.shape_stride.total_size();
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
            if self.shape_stride.dims() != index.len() {
                panic!("Invalid index (dimension count mismatch)");
            }

            let mut offset = 0;
            for (i, item) in index.iter().enumerate() {
                offset += item * self.shape_stride.stride(i);
            }
            storage[offset]
        } else {
            panic!("Cannot extract f32 value from non-float32 tensor");
        }
    }

    pub fn shape(&self) -> Shape {
        self.shape_stride.shape.clone()
    }

    pub fn add(&self, rhs: &Self) -> Self {
        // Make sure the shape and the type match.
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Incompatible tensors - different shapes"
        );
        assert_eq!(
            self.dtype(),
            rhs.dtype(),
            "Incompatible tensors - different types"
        );

        match (self.storage.as_ref(), rhs.storage.as_ref()) {
            (TensorStorage::Float32(lhs_storage), TensorStorage::Float32(rhs_storage)) => {
                let (shape, result) = pointwise_binop::<f32, _>(
                    lhs_storage,
                    &self.shape_stride,
                    rhs_storage,
                    &rhs.shape_stride,
                    f32::add,
                    0.0f32,
                );
                Tensor {
                    shape_stride: shape,
                    storage: Arc::new(TensorStorage::Float32(result)),
                }
            }
            (TensorStorage::Int32(lhs_storage), TensorStorage::Int32(rhs_storage)) => {
                let (shape, result) = pointwise_binop::<i32, _>(
                    lhs_storage,
                    &self.shape_stride,
                    rhs_storage,
                    &rhs.shape_stride,
                    i32::add,
                    0,
                );
                Tensor {
                    shape_stride: shape,
                    storage: Arc::new(TensorStorage::Int32(result)),
                }
            }
            (_, _) => panic!("Unsupported combination for addition"),
        }
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        // Make sure the shape and the type match.
        assert_eq!(
            self.shape_stride, rhs.shape_stride,
            "Incompatible tensors - different shapes"
        );
        assert_eq!(
            self.dtype(),
            rhs.dtype(),
            "Incompatible tensors - different types"
        );

        match (self.storage.as_ref(), rhs.storage.as_ref()) {
            (TensorStorage::Float32(lhs_storage), TensorStorage::Float32(rhs_storage)) => {
                let (shape, result) = pointwise_binop::<f32, _>(
                    lhs_storage,
                    &self.shape_stride,
                    rhs_storage,
                    &rhs.shape_stride,
                    f32::mul,
                    0.0f32,
                );
                Tensor {
                    shape_stride: shape,
                    storage: Arc::new(TensorStorage::Float32(result)),
                }
            }
            (TensorStorage::Int32(lhs_storage), TensorStorage::Int32(rhs_storage)) => {
                let (shape, result) = pointwise_binop::<i32, _>(
                    lhs_storage,
                    &self.shape_stride,
                    rhs_storage,
                    &rhs.shape_stride,
                    i32::mul,
                    0,
                );
                Tensor {
                    shape_stride: shape,
                    storage: Arc::new(TensorStorage::Int32(result)),
                }
            }
            (_, _) => panic!("Unsupported combination for addition"),
        }
    }

    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape_stride.dims(),
            other.shape_stride.dims(),
            "matmul requires tensor of same dimensions"
        );
        let d = self.shape_stride.dims();
        assert!(
            d >= 2,
            "matmul operands must have at least two dimensions, left operand has {}",
            d
        );
        assert!(
            other.shape_stride.dims() >= 2,
            "matmul operands must have at least two dimensions, right operand has {}",
            other.shape_stride.dims()
        );
        for i in 0..d - 2 {
            assert_eq!(
                self.shape_stride.size(i),
                other.shape_stride.size(i),
                "matmul requires same size of dimensions [0..dims-2]"
            );
        }
        let m = self.shape_stride.size(d - 2);
        let n = other.shape_stride.size(d - 1);
        let k = self.shape_stride.size(d - 1);
        assert_eq!(
            k,
            other.shape_stride.size(d - 2),
            "matmul requires matching lhs[d-1] and rhs[d-2] dimension sizes"
        );
        match (self.storage.as_ref(), other.storage.as_ref()) {
            (TensorStorage::Float32(lhs_storage), TensorStorage::Float32(rhs_storage)) => {
                // Build the result storage and the result shape.
                let mut result_shape_builder = ContiguousShapeBuilder::new();
                result_shape_builder.add_low(n);
                result_shape_builder.add_low(m);
                for i in 0..d - 2 {
                    let dim_size = self.shape_stride.size(i);
                    result_shape_builder.add_low(dim_size);
                }
                let result_size = result_shape_builder.size();
                let mut result_storage = vec![0.0f32; result_size];
                let result_shape = result_shape_builder.finish();
                let mut result_offset = 0usize;
                let mut lhs_tensor_index = TensorIndex::from_shape(&self.shape_stride, d - 2);
                let mut rhs_tensor_index = TensorIndex::from_shape(&other.shape_stride, d - 2);

                let lhs_stride = self.shape_stride.dim(d - 1).stride;
                let rhs_stride = other.shape_stride.dim(d - 2).stride;

                while result_offset < result_size {
                    let lhs_offset = lhs_tensor_index.next().unwrap();
                    let rhs_offset = rhs_tensor_index.next().unwrap();

                    for i in 0..m {
                        let lhs_index = lhs_offset + i * self.shape_stride.dim(d - 2).stride;
                        for j in 0..n {
                            let rhs_index = rhs_offset + j * other.shape_stride.dim(d - 1).stride;
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += lhs_storage[lhs_index + l * lhs_stride]
                                    * rhs_storage[rhs_index + l * rhs_stride];
                            }
                            result_storage[result_offset] = sum;
                            result_offset += 1;
                        }
                    }
                }
                Self {
                    shape_stride: result_shape,
                    storage: Arc::new(TensorStorage::Float32(result_storage)),
                }
            }
            (_, _) => panic!("Only float32 matrix multiplication is supported"),
        }
    }

    // Transposes the last two axes.
    pub fn transpose(&self) -> Self {
        let mut shape_dims = self.shape_stride.to_shape_dimensions();

        let d = shape_dims.len();
        shape_dims.swap(d - 2, d - 1);
        Self {
            shape_stride: ShapeStride::from(shape_dims),
            storage: self.storage.clone(),
        }
    }

    pub fn reshape(&self, shape: Shape) -> Self {
        let size = self.shape_stride.total_size();
        let new_size = shape.as_ref().iter().product();
        assert_eq!(
            size, new_size,
            "Tensor can be only reshaped to same size tensors."
        );
        // If the current shape is continuous, then just reuse the same storage with the new shape.
        let new_shape = ContiguousShapeBuilder::from_sizes(shape.as_ref()).finish();
        if self.shape_stride.is_contiguous() {
            return Self {
                shape_stride: new_shape,
                storage: self.storage.clone(),
            };
        }

        // Otherwise we need to copy.
        let dims = self.shape_stride.dims();
        let mut index = TensorIndex::from_shape(&self.shape_stride, dims - 1);
        let ShapeDimension { size, stride } = if dims > 0 {
            self.shape_stride.dim(dims - 1)
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
                    shape_stride: new_shape,
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
                    shape_stride: new_shape,
                }
            }
        }
    }

    pub fn broadcast(&self, dim_size_pairs: &[(usize, usize)]) -> Self {
        let mut shape_dims = self.shape_stride.to_shape_dimensions();
        let mut last_dim: Option<usize> = None;
        for (dim, size) in dim_size_pairs {
            assert!(
                last_dim.is_none() || last_dim.unwrap() < *dim,
                "Broadcast (dimension, size) pairs must be sorted by dimension index"
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
            shape_stride: ShapeStride::from(shape_dims),
            storage: self.storage.clone(),
        }
    }

    pub fn sum(&self, axis: Option<&[usize]>, keep_dim: bool) -> Self {
        let dims = self.shape_stride.dims();
        let mut other_index = TensorIndex::new();
        let mut sum_index;
        let mut shape_builder = ContiguousShapeBuilder::new();
        if let Some(axis) = axis {
            sum_index = TensorIndex::new();
            let mut finger = axis.len();
            for i in (0..dims).rev() {
                if finger > 0 && axis[finger - 1] == i {
                    finger -= 1;
                    sum_index.add_low(self.shape_stride.dim(i));
                    if keep_dim {
                        shape_builder.add_low(1);
                    }
                } else {
                    other_index.add_low(self.shape_stride.dim(i));
                    shape_builder.add_low(self.shape_stride.dim(i).size);
                };
            }
        } else {
            sum_index = TensorIndex::from_shape(&self.shape_stride, dims);
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
                    shape_stride: shape_builder.finish(),
                    storage: TensorStorage::Float32(result_storage).into(),
                }
            }
            _ => panic!("Sum is only supported for f32 tensors"),
        }
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<")?;
        match self.storage.as_ref() {
            TensorStorage::Float32(storage) => write!(f, "{:?}", storage)?,
            TensorStorage::Int32(storage) => write!(f, "{:?}", storage)?,
        }
        write!(f, ", shape: {:?}", self.shape_stride.shape)?;
        write!(f, ", strides: {:?}", self.shape_stride.strides)?;
        write!(f, ">")?;
        Ok(())
    }
}

fn pointwise_binop<T: Copy, F>(
    lhs: &[T],
    lhs_shape: &ShapeStride,
    rhs: &[T],
    rhs_shape: &ShapeStride,
    op: F,
    init: T,
) -> (ShapeStride, Vec<T>)
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
    for size in lhs_shape.shape.as_ref().iter().rev() {
        result_shape_builder.add_low(*size);
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
    pub fn from_shape(shape: &ShapeStride, dimensions: usize) -> Self {
        let mut res = Self::with_capacity(dimensions);
        let shape_dims = shape.to_shape_dimensions();
        for dim in shape_dims[..usize::min(dimensions, shape_dims.len())]
            .iter()
            .rev()
        {
            res.add_low(*dim);
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
        for s in sizes.iter().rev() {
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

    fn finish(self) -> ShapeStride {
        let mut dims = self.dims;
        dims.reverse();
        ShapeStride::from(dims)
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
    assert_eq!(&p.shape().as_ref(), &[1, 1]);
}

#[test]
fn matmul_2x3_3x1() {
    let t1 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let t2 = Tensor::from_data_f32(&[4.0, 3.0, 2.0], &[1, 3]).transpose();
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_data_f32(), &[16.0, 43.0]);
    assert_eq!(&p.shape().as_ref(), &[2, 1]);
}

#[test]
fn matmul_2x_1x3_3x1() {
    let t1 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    let t2 = Tensor::from_data_f32(&[4.0, 3.0, 2.0, 1.0, 2.0, 3.0], &[2, 1, 3]).transpose();
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_data_f32(), &[16.0, 32.0]);
    assert_eq!(&p.shape().as_ref(), &[2, 1, 1]);
}

#[test]
fn matmul_2x_1x2_2x2() {
    let t1 = Tensor::from_data_f32(&[4.0, 3.0, 2.0, 1.0], &[2, 1, 2]);
    let t2 = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_data_f32(), &[13.0, 20.0, 17.0, 20.0]);
    assert_eq!(&p.shape().as_ref(), &[2, 1, 2]);
}

#[test]
fn matmul_2x_1x2_2x2_t() {
    let t1 = Tensor::from_data_f32(&[4.0, 3.0, 2.0, 1.0], &[2, 1, 2]);
    let t2 =
        Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]).transpose();
    let p = t1.matmul(&t2);
    assert_eq!(&p.to_data_f32(), &[10.0, 24.0, 16.0, 22.0]);
    assert_eq!(&p.shape().as_ref(), &[2, 1, 2]);
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
    assert_eq!(&p.shape().as_ref(), &[2, 3, 2, 1]);
}

#[test]
fn reshape_2x2_to_4() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let r = t.reshape(Shape::from_iter([4]));
    assert_eq!(&r.to_data_f32(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&r.shape().as_ref(), &[4]);
}

#[test]
fn reshape_2_to_2x1() {
    let t = Tensor::from_data_f32(&[1.0, 2.0], &[2]);
    let r = t.reshape(Shape::from_iter([2, 1]));
    assert_eq!(&r.to_data_f32(), &[1.0, 2.0]);
    assert_eq!(&r.shape().as_ref(), &[2, 1]);
}

#[test]
fn reshape_1x1_to_scalar() {
    let t = Tensor::from_data_f32(&[1.0], &[1, 1]);
    let r = t.reshape(Shape::from_iter([]));
    assert_eq!(&r.to_data_f32(), &[1.0]);
    assert_eq!(&r.shape().as_ref(), &[]);
}

#[test]
fn reshape_3x2_t_to_6() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).transpose();
    let r = t.reshape(Shape::from_iter([6]));
    assert_eq!(&r.to_data_f32(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
    assert_eq!(&r.shape().as_ref(), &[6]);
}

#[test]
fn ones_3x2() {
    let t = Tensor::ones(Shape::from_iter([3, 2]));
    assert_eq!(&t.to_data_f32(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&t.shape().as_ref(), &[3, 2]);
}

#[test]
fn sum_3x2_to_2() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = t.sum(Some(&[1]), false);
    assert_eq!(&s.to_data_f32(), &[6.0, 15.0]);
    assert_eq!(&s.shape().as_ref(), &[2]);
}

#[test]
fn sum_3x2_to_3() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = t.sum(Some(&[0]), false);
    assert_eq!(&s.to_data_f32(), &[5.0, 7.0, 9.0]);
    assert_eq!(&s.shape().as_ref(), &[3]);
}

#[test]
fn sum_3x2_to_2_keepdim() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let s = t.sum(Some(&[1]), true);
    assert_eq!(&s.to_data_f32(), &[6.0, 15.0]);
    assert_eq!(&s.shape().as_ref(), &[2, 1]);
}

#[test]
fn sum_2x2x2_to_2_keepdim() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let s = t.sum(Some(&[1]), true);
    assert_eq!(&s.to_data_f32(), &[4.0, 6.0, 12.0, 14.0]);
    assert_eq!(&s.shape().as_ref(), &[2, 1, 2]);
}

#[test]
fn sum_3x2_t_to_3() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).transpose();
    let s = t.sum(Some(&[1]), false);
    assert_eq!(&s.to_data_f32(), &[5.0, 7.0, 9.0]);
    assert_eq!(&s.shape().as_ref(), &[3]);
}

#[test]
fn sum_2x2_to_scalar() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let s = t.sum(None, false);
    assert_eq!(&s.to_data_f32(), &[10.0]);
    assert_eq!(&s.shape().as_ref(), &[]);
}

#[test]
fn sum_2x2_to_scalar_keepdim() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let s = t.sum(None, true);
    assert_eq!(&s.to_data_f32(), &[10.0]);
    assert_eq!(&s.shape().as_ref(), &[1, 1]);
}

#[test]
fn broadcast_1d() {
    let t = Tensor::from_data_f32(&[1.0], &[1]);
    let r = t.broadcast(&[(0, 4)]);
    assert_eq!(&r.to_data_f32(), &[1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&r.shape().as_ref(), &[4]);
}

#[test]
fn broadcast_1x1_to_3x2() {
    let t = Tensor::from_data_f32(&[1.0], &[1, 1]);
    let r = t.broadcast(&[(0, 2), (1, 3)]);
    assert_eq!(&r.to_data_f32(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    assert_eq!(&r.shape().as_ref(), &[2, 3]);
}
#[test]

fn broadcast_2x1x2_to_2x2x2() {
    let t = Tensor::from_data_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let r = t.broadcast(&[(1, 2)]);
    assert_eq!(&r.to_data_f32(), &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    assert_eq!(&r.shape().as_ref(), &[2, 2, 2]);
}
