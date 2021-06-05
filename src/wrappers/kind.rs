//! The different kind of elements supported in Torch.

use half::f16;

/// The different kind of elements that a Tensor can hold.
///
/// [Element] can be used for a value of a specific kind.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Kind {
    Uint8,
    Int8,
    Int16,
    Int,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
}

impl Kind {
    pub(super) fn c_int(self) -> libc::c_int {
        // These values should be in sync with include/c10/core/ScalarType.h
        match self {
            Kind::Uint8 => 0,
            Kind::Int8 => 1,
            Kind::Int16 => 2,
            Kind::Int => 3,
            Kind::Int64 => 4,
            Kind::Half => 5,
            Kind::Float => 6,
            Kind::Double => 7,
            Kind::ComplexHalf => 8,
            Kind::ComplexFloat => 9,
            Kind::ComplexDouble => 10,
            Kind::Bool => 11,
            Kind::QInt8 => 12,
            Kind::QUInt8 => 13,
            Kind::QInt32 => 14,
            Kind::BFloat16 => 15,
        }
    }

    pub(super) fn of_c_int(v: libc::c_int) -> Result<Kind, crate::TchError> {
        match v {
            0 => Ok(Kind::Uint8),
            1 => Ok(Kind::Int8),
            2 => Ok(Kind::Int16),
            3 => Ok(Kind::Int),
            4 => Ok(Kind::Int64),
            5 => Ok(Kind::Half),
            6 => Ok(Kind::Float),
            7 => Ok(Kind::Double),
            8 => Ok(Kind::ComplexHalf),
            9 => Ok(Kind::ComplexFloat),
            10 => Ok(Kind::ComplexDouble),
            11 => Ok(Kind::Bool),
            12 => Ok(Kind::QInt8),
            13 => Ok(Kind::QUInt8),
            14 => Ok(Kind::QInt32),
            15 => Ok(Kind::BFloat16),
            _ => Err(crate::TchError::UnknownKind(v)),
        }
    }

    pub fn elt_size_in_bytes(self) -> usize {
        match self {
            Kind::Uint8 => 1,
            Kind::Int8 => 1,
            Kind::Int16 => 2,
            Kind::Int => 4,
            Kind::Int64 => 8,
            Kind::Half => 2,
            Kind::Float => 4,
            Kind::Double => 8,
            Kind::ComplexHalf => 4,
            Kind::ComplexFloat => 8,
            Kind::ComplexDouble => 16,
            Kind::Bool => 1,
            Kind::QInt8 => 1,
            Kind::QUInt8 => 1,
            Kind::QInt32 => 4,
            Kind::BFloat16 => 2,
        }
    }
}

pub const FLOAT_CPU: (Kind, crate::Device) = (Kind::Float, crate::Device::Cpu);
pub const DOUBLE_CPU: (Kind, crate::Device) = (Kind::Double, crate::Device::Cpu);
pub const INT64_CPU: (Kind, crate::Device) = (Kind::Int64, crate::Device::Cpu);

pub const FLOAT_CUDA: (Kind, crate::Device) = (Kind::Float, crate::Device::Cuda(0));
pub const DOUBLE_CUDA: (Kind, crate::Device) = (Kind::Double, crate::Device::Cuda(0));
pub const INT64_CUDA: (Kind, crate::Device) = (Kind::Int64, crate::Device::Cuda(0));

/// Single element of a specific [Kind].
///
/// # Implementation
///
/// The size of the type should be the same as corresponding [Kind::elt_size_in_bytes]
pub unsafe trait Element: Copy {
    const KIND: Kind;
}

unsafe impl Element for u8 {
    const KIND: Kind = Kind::Uint8;
}

unsafe impl Element for i8 {
    const KIND: Kind = Kind::Int8;
}

unsafe impl Element for i16 {
    const KIND: Kind = Kind::Int16;
}

unsafe impl Element for i32 {
    const KIND: Kind = Kind::Int;
}

unsafe impl Element for i64 {
    const KIND: Kind = Kind::Int64;
}

unsafe impl Element for f16 {
    const KIND: Kind = Kind::Half;
}

unsafe impl Element for f32 {
    const KIND: Kind = Kind::Float;
}

unsafe impl Element for f64 {
    const KIND: Kind = Kind::Double;
}

unsafe impl Element for bool {
    const KIND: Kind = Kind::Bool;
}
