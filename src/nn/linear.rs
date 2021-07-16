//! A linear fully-connected layer.
use crate::wrappers::kind::Kind::Float;
use crate::Tensor;
use std::borrow::Borrow;

/// Configuration for a linear layer.
#[derive(Debug, Clone, Copy)]
pub struct LinearConfig {
    pub ws_init: super::Init,
    pub bs_init: Option<super::Init>,
    pub bias: bool,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            ws_init: super::Init::KaimingUniform,
            bs_init: None,
            bias: true,
        }
    }
}

/// A linear fully-connected layer.
#[derive(Debug)]
pub struct Linear {
    pub ws: Tensor,
    pub bs: Tensor,
}

/// Creates a new linear layer.
pub fn linear<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    in_dim: i64,
    out_dim: i64,
    c: LinearConfig,
) -> Linear {
    let vs = vs.borrow();
    let bs = if c.bias {
        let bs_init = c.bs_init.unwrap_or_else(|| {
            let bound = 1.0 / (in_dim as f64).sqrt();
            super::Init::Uniform {
                lo: -bound,
                up: bound,
            }
        });
        vs.var("bias", &[out_dim], bs_init)
    } else {
        Tensor::zeros(&[out_dim], (Float, vs.device()))
    };

    Linear {
        ws: vs.var("weight", &[out_dim, in_dim], c.ws_init),
        bs,
    }
}

impl super::module::Module for Linear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.matmul(&self.ws.tr()) + &self.bs
    }
}
