//! A batch-normalization layer.
use crate::Tensor;
use std::borrow::Borrow;

/// Batch-normalization config.
#[derive(Debug, Clone, Copy)]
pub struct BatchNormConfig {
    pub cudnn_enabled: bool,
    pub eps: f64,
    pub momentum: f64,
    pub ws_init: super::Init,
    pub bs_init: super::Init,
}

impl Default for BatchNormConfig {
    fn default() -> Self {
        Self {
            cudnn_enabled: true,
            eps: 1e-5,
            momentum: 0.1,
            ws_init: super::Init::Uniform { lo: 0., up: 1. },
            bs_init: super::Init::Const(0.),
        }
    }
}

/// A batch-normalization layer.
#[derive(Debug)]
pub struct BatchNorm {
    config: BatchNormConfig,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub ws: Tensor,
    pub bs: Tensor,
    pub nd: usize,
}

fn batch_norm<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    nd: usize,
    out_dim: i64,
    config: BatchNormConfig,
) -> BatchNorm {
    let vs = vs.borrow();
    BatchNorm {
        config,
        running_mean: vs.zeros_no_train("running_mean", &[out_dim]),
        running_var: vs.ones_no_train("running_var", &[out_dim]),
        ws: vs.var("weight", &[out_dim], config.ws_init),
        bs: vs.var("bias", &[out_dim], config.bs_init),
        nd,
    }
}

/// Applies Batch Normalization over a three dimension input.
///
/// The input shape is assumed to be (N, C, L). Normalization
/// is performed over the first batch dimension N.
pub fn batch_norm1d<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    out_dim: i64,
    config: BatchNormConfig,
) -> BatchNorm {
    batch_norm(vs, 1, out_dim, config)
}

/// Applies Batch Normalization over a four dimension input.
///
/// The input shape is assumed to be (N, C, H, W). Normalization
/// is performed over the first batch dimension N.
pub fn batch_norm2d<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    out_dim: i64,
    config: BatchNormConfig,
) -> BatchNorm {
    batch_norm(vs, 2, out_dim, config)
}

/// Applies Batch Normalization over a five dimension input.
///
/// The input shape is assumed to be (N, C, D, H, W). Normalization
/// is performed over the first batch dimension N.
pub fn batch_norm3d<'a, T: Borrow<super::Path<'a>>>(
    vs: T,
    out_dim: i64,
    config: BatchNormConfig,
) -> BatchNorm {
    batch_norm(vs, 3, out_dim, config)
}

impl super::module::ModuleT for BatchNorm {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let dim = xs.dim();
        if self.nd == 1 && dim != 2 && dim != 3 {
            panic!(
                "expected an input tensor with 2 or 3 dims, got {:?}",
                xs.size()
            )
        }
        if self.nd > 1 && xs.dim() != self.nd + 2 {
            panic!(
                "expected an input tensor with {} dims, got {:?}",
                self.nd + 2,
                xs.size()
            )
        };
        Tensor::batch_norm(
            xs,
            Some(&self.ws),
            Some(&self.bs),
            Some(&self.running_mean),
            Some(&self.running_var),
            train,
            self.config.momentum,
            self.config.eps,
            self.config.cudnn_enabled,
        )
    }
}
