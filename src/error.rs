use std::error;
use std::fmt;

pub type Result<T> = std::result::Result<T, TensorError>;

#[derive(Debug, Clone)]
pub struct TensorError(String);

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Invalid op: {}", self.0)
    }
}

impl error::Error for TensorError {}

impl TensorError {
    pub fn new(message: &str) -> Self {
        Self(message.to_string())
    }
}