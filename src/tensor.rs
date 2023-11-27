use std::cmp::PartialEq;
use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::sync::Arc;

use crate::error::{Result, TensorError};

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

    pub fn get_f32_item(&self, index: &[usize]) -> Result<f32> {
        if let TensorStorage::Float32(storage) = self.storage.as_ref() {
            if self.shape.dims() != index.len() {
                return Err(TensorError::new("Invalid index (dimension count mismatch"));
            }

            let mut offset = 0;
            for i in 0..index.len() {
                offset += index[i] * self.shape.dim(i).stride;
            }
            Ok(storage[offset])
        } else {
            Err(TensorError::new(
                "Cannot extract f32 value from non-float32 tensor",
            ))
        }
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
    if d == 0 {
        return (lhs_shape.clone(), vec![op(lhs[0], rhs[0])]);
    }
    let mut iteration_space = vec![(0usize, 0usize, 0usize); d];
    let ShapeDimension {
        size: batch_size,
        stride: lhs_stride,
    } = lhs_shape.dim(d - 1);
    let rhs_stride = rhs_shape.dim(d - 1).stride;

    // Currently, we just adopt the LHS storage, but that might not be the smartest idea.
    let mut size = 1usize;
    let mut res_shape = vec![ShapeDimension { size: 0, stride: 0 }; d];
    for i in (0..lhs_shape.dims()).rev() {
        let dim_size = lhs_shape.dim(i).size;
        res_shape[i] = ShapeDimension {
            size: dim_size,
            stride: size,
        };
        size *= dim_size;
    }
    let mut result_storage = vec![init; size];
    let mut res_pos = 0usize;
    'outer: loop {
        // Get the start of the iteration.
        let (mut lhs_pos, mut rhs_pos) = if d >= 2 {
            (iteration_space[d - 2].1, iteration_space[d - 2].2)
        } else {
            (0usize, 0usize)
        };
        for _i in 0..batch_size {
            result_storage[res_pos] = op(lhs[lhs_pos], rhs[rhs_pos]);
            lhs_pos += lhs_stride;
            rhs_pos += rhs_stride;
            res_pos += 1;
        }

        // Move the finger forwards.
        let mut current = d - 1;
        loop {
            if current == 0 {
                // There is no dimension to advance, we are done now.
                break 'outer;
            }
            current -= 1;
            let next_index = iteration_space[current].0 + 1;
            if next_index != lhs_shape.dim(current).size {
                // Bump the pointers for this dimension.
                // We also need to reset the pointers in all the higher dimensions,
                // but we do it just once outside of this loop.
                iteration_space[current].0 = next_index;
                iteration_space[current].1 += lhs_shape.dim(current).stride;
                iteration_space[current].2 += rhs_shape.dim(current).stride;
                break;
            }
            // We reached the end of the |current| dimension.
            // Let us try to move to th eouter dimension.
        }
        // Reset all the higher dimension pointers.
        for i in current + 1..d {
            iteration_space[i] = (0, iteration_space[i - 1].1, iteration_space[i - 1].2);
        }
        println!("Bumped: {:?}", iteration_space);
    }
    (Shape(res_shape), result_storage)
}

#[test]
fn scalar_add() {
    let t = Tensor::scalar_f32(1.0).add(&Tensor::scalar_f32(2.0));
    assert_eq!(t.get_f32_item(&[]).unwrap(), 3.0);
}

#[test]
fn scalar_mul() {
    let t = Tensor::scalar_f32(2.0).mul(&Tensor::scalar_f32(3.0));
    assert_eq!(t.get_f32_item(&[]).unwrap(), 6.0);
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
