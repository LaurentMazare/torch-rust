use anyhow::{bail, Result};
use std::{borrow::Borrow, path::Path};
use tch::{
    kind::{Kind, INT64_CPU},
    nn::{self, OptimizerConfig},
    Device, Tensor,
};

// model parameters
const IMAGE_DIM: i64 = 3;
const LATENT_DIM: i64 = 128;
const IMAGE_WIDTH: i64 = 64;

// training parameters
const BATCH_SIZE: i64 = 32;
const LEARNING_RATE: f64 = 1e-4;
const DIS_ITERATIONS: i64 = 5;
const GP_LAMBDA: f64 = 10.0;
const MAX_STEPS: i64 = 100000000;

// logging parameters
const N_IMAGES: i64 = 8;
const SAVE_IMAGE_PER_STEPS: i64 = 100;

fn main() -> Result<()> {
    // load images for training
    let args: Vec<_> = std::env::args().collect();
    let (image_dir, output_dir) = match args.as_slice() {
        [_, image_dir, output_dir] => (Path::new(image_dir), Path::new(output_dir)),
        [prog_name, ..] => bail!("usage: {} IMAGE_DATASET_DIR OUTPUT_DIR", prog_name),
        [] => bail!("usage: main IMAGE_DATASET_DIR OUTPUT_DIR"),
    };
    let images = load_image_dir(image_dir, IMAGE_WIDTH)?;
    let train_size = images.size()[0];
    println!("loaded {} images", train_size);

    // prerequisites
    let device = Device::cuda_if_available();
    std::fs::create_dir_all(&output_dir)?;

    // utility functions
    let rand_latent = |bs| {
        nn::init(
            nn::Init::Uniform { lo: -1., up: 1. },
            &[bs, LATENT_DIM, 1, 1],
            device,
        )
    };
    let random_batch_images = |bs| {
        let index = Tensor::randint(train_size, &[bs], INT64_CPU);
        images.index_select(0, &index).to_device(device)
    };
    let fixed_noise = rand_latent(N_IMAGES * N_IMAGES);

    // create model and optimizers
    let mut mod_gen_vs = nn::VarStore::new(device);
    let mod_gen = model::generator(mod_gen_vs.root(), LATENT_DIM, IMAGE_DIM);
    let mut opt_gen = nn::adam(0.5, 0.999, 0.).build(&mod_gen_vs, LEARNING_RATE)?;

    let mut mod_dis_vs = nn::VarStore::new(device);
    let mod_dis = model::discriminator(mod_dis_vs.root(), IMAGE_DIM);
    let mut opt_dis = nn::adam(0.5, 0.999, 0.).build(&mod_dis_vs, LEARNING_RATE)?;

    for step in 0..MAX_STEPS {
        // Training of discriminator(more iterations than the generator).
        let loss_dis = (0..DIS_ITERATIONS)
            .map(|_| {
                mod_dis_vs.unfreeze();
                mod_gen_vs.freeze();

                let loss_dis = {
                    let real_images = random_batch_images(BATCH_SIZE);
                    let fake_images = rand_latent(BATCH_SIZE).apply_t(&mod_gen, true);
                    let y_pred_real = real_images.apply_t(&mod_dis, true);
                    let y_pred_fake = fake_images.apply_t(&mod_dis, true);

                    let wdist = (&y_pred_real - &y_pred_fake).mean(Kind::Float);
                    let gp = model::gradient_penalty(
                        &mod_dis,
                        GP_LAMBDA,
                        &real_images,
                        &fake_images,
                        true,
                    );

                    wdist + gp
                };
                opt_dis.backward_step(&loss_dis);
                f32::from(loss_dis)
            })
            .sum::<f32>()
            / DIS_ITERATIONS as f32;

        // Training of generator.
        let loss_gen = {
            mod_dis_vs.freeze();
            mod_gen_vs.unfreeze();

            let loss_gen = {
                let batch_images = random_batch_images(BATCH_SIZE);
                let y_pred_real = batch_images.apply_t(&mod_dis, true);
                let y_pred_fake = rand_latent(BATCH_SIZE)
                    .apply_t(&mod_gen, true)
                    .apply_t(&mod_dis, true);
                let neg_wdist = (&y_pred_fake - &y_pred_real).mean(Kind::Float);
                neg_wdist
            };
            opt_gen.backward_step(&loss_gen);
            f32::from(loss_gen)
        };

        println!("step: {},\tD: {:?},\tG: {:?}", step, loss_dis, loss_gen);

        if step % SAVE_IMAGE_PER_STEPS == 0 {
            let imgs = fixed_noise
                .apply_t(&mod_gen, false)
                .view([-1, IMAGE_DIM, IMAGE_WIDTH, IMAGE_WIDTH])
                .to_device(Device::Cpu);
            tch::vision::image::save(
                &image_matrix(&imgs, N_IMAGES)?,
                output_dir.join(format!("relout{}.png", step)),
            )?;
        }
    }

    Ok(())
}

fn image_matrix(imgs: &Tensor, sz: i64) -> Result<Tensor> {
    let imgs = ((imgs + 1.) * 127.5).clamp(0., 255.).to_kind(Kind::Uint8);
    let mut ys: Vec<Tensor> = vec![];
    for i in 0..sz {
        ys.push(Tensor::cat(
            &(0..sz)
                .map(|j| imgs.narrow(0, sz * i + j, 1))
                .collect::<Vec<_>>(),
            2,
        ))
    }
    Ok(Tensor::cat(&ys, 3).squeeze1(0))
}

fn load_image_dir<P>(image_dir: P, image_width: i64) -> Result<Tensor>
where
    P: AsRef<Path>,
{
    let images = tch::vision::image::load_dir(image_dir.as_ref(), image_width, image_width)?;
    Ok(images.to_kind(Kind::Float) / 127.5 - 1.0)
}

mod model {
    use super::*;

    pub fn tr2d<'p, P>(
        p: P,
        c_in: i64,
        c_out: i64,
        padding: i64,
        stride: i64,
    ) -> nn::ConvTranspose2D
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        let cfg = nn::ConvTransposeConfig {
            stride,
            padding,
            bias: false,
            ..Default::default()
        };
        nn::conv_transpose2d(p, c_in, c_out, 4, cfg)
    }

    pub fn conv2d<'p, P>(p: P, c_in: i64, c_out: i64, padding: i64, stride: i64) -> nn::Conv2D
    where
        P: Borrow<nn::Path<'p>>,
    {
        let cfg = nn::ConvConfig {
            stride,
            padding,
            bias: false,
            ..Default::default()
        };
        nn::conv2d(p, c_in, c_out, 4, cfg)
    }

    pub fn generator<'p, P>(p: P, latent_dim: i64, output_dim: i64) -> nn::SequentialT
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        nn::seq_t()
            .add(tr2d(p / "tr1", latent_dim, 1024, 0, 1))
            .add(nn::batch_norm2d(p / "bn1", 1024, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(tr2d(p / "tr2", 1024, 512, 1, 2))
            .add(nn::batch_norm2d(p / "bn2", 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(tr2d(p / "tr3", 512, 256, 1, 2))
            .add(nn::batch_norm2d(p / "bn3", 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(tr2d(p / "tr4", 256, 128, 1, 2))
            .add(nn::batch_norm2d(p / "bn4", 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(tr2d(p / "tr5", 128, output_dim, 1, 2))
            .add_fn(|xs| xs.tanh())
    }

    pub fn discriminator<'p, P>(p: P, input_dim: i64) -> nn::SequentialT
    where
        P: Borrow<nn::Path<'p>>,
    {
        let p = p.borrow();
        nn::seq_t()
            .add(conv2d(p / "conv1", input_dim, 128, 1, 2))
            .add_fn(leaky_relu)
            .add(conv2d(p / "conv2", 128, 256, 1, 2))
            .add(instance_norm2d(p / "in2", 256, Default::default()))
            .add_fn(leaky_relu)
            .add(conv2d(p / "conv3", 256, 512, 1, 2))
            .add(instance_norm2d(p / "in3", 512, Default::default()))
            .add_fn(leaky_relu)
            .add(conv2d(p / "conv4", 512, 1024, 1, 2))
            .add(instance_norm2d(p / "in4", 1024, Default::default()))
            .add_fn(leaky_relu)
            .add(conv2d(p / "conv5", 1024, 1, 0, 1))
    }

    pub struct InstanceNormConfig {
        pub ws_init: nn::Init,
        pub bs_init: nn::Init,
        pub momentum: f64,
        pub eps: f64,
        pub cudnn_enabled: bool,
    }

    impl Default for InstanceNormConfig {
        fn default() -> Self {
            Self {
                cudnn_enabled: true,
                eps: 1e-5,
                momentum: 0.1,
                ws_init: nn::Init::Uniform { lo: 0., up: 1. },
                bs_init: nn::Init::Const(0.),
            }
        }
    }

    pub fn instance_norm2d<'p, P>(
        path: P,
        out_dim: i64,
        config: InstanceNormConfig,
    ) -> nn::FuncT<'static>
    where
        P: Borrow<nn::Path<'p>>,
    {
        instance_norm(path, 2, out_dim, config)
    }

    pub fn instance_norm<'p, P>(
        path: P,
        num_data_dims: i64,
        out_dim: i64,
        config: InstanceNormConfig,
    ) -> nn::FuncT<'static>
    where
        P: Borrow<nn::Path<'p>>,
    {
        let path = path.borrow();
        let InstanceNormConfig {
            ws_init,
            bs_init,
            momentum,
            eps,
            cudnn_enabled,
        } = config;

        let running_mean = path.zeros_no_train("running_mean", &[out_dim]);
        let running_var = path.ones_no_train("running_var", &[out_dim]);
        let weight = path.var("weight", &[out_dim], ws_init);
        let bias = path.var("bias", &[out_dim], bs_init);

        nn::func_t(move |xs, train| {
            // sanity check
            let input_dims = xs.dim() as i64;
            if num_data_dims == 1 && ![2, 3].contains(&input_dims) {
                panic!(
                    "expected an input tensor with 2 or 3 dims, got {:?}",
                    xs.size()
                )
            }
            if num_data_dims > 1 && input_dims != num_data_dims + 2 {
                panic!(
                    "expected an input tensor with {} dims, got {:?}",
                    num_data_dims + 2,
                    xs.size()
                )
            };

            // run instance norm
            Tensor::instance_norm(
                xs,
                Some(&weight),
                Some(&bias),
                Some(&running_mean),
                Some(&running_var),
                train,
                momentum,
                eps,
                cudnn_enabled,
            )
        })
    }

    pub fn leaky_relu(xs: &Tensor) -> Tensor {
        xs.max1(&(xs * 0.2))
    }

    pub fn gradient_penalty<M>(
        discriminator: &M,
        lambda: f64,
        real: &Tensor,
        fake: &Tensor,
        train: bool,
    ) -> Tensor
    where
        M: nn::ModuleT,
    {
        debug_assert_eq!(real.device(), fake.device());
        debug_assert_eq!(real.kind(), fake.kind());
        debug_assert_eq!(real.size(), fake.size());
        debug_assert_eq!(real.size().len(), 4);

        let device = real.device();
        let kind = real.kind();
        let shape = real.size();
        let bsize = shape[0];

        let ratios = nn::init(
            nn::Init::Uniform { lo: 0.0, up: 1.0 },
            &[bsize, 1, 1, 1],
            device,
        )
        .expand_as(&real);

        let interpolated: Tensor = &ratios * real + (1.0 - ratios) * fake;
        let interpolated = interpolated.set_requires_grad(true);

        let discriminator_score = discriminator.forward_t(&interpolated, train);

        let gradients = &Tensor::run_backward(
            &[&discriminator_score], // outputs
            &[&interpolated],        // inputs
            true,                    // keep_graph
            true,                    // create_graph
        )[0];

        let gradient_penalty = (gradients.norm2(2, &[1, 2, 3], false) - 1.0)
            .pow(2.0)
            .mean(kind)
            * lambda;
        gradient_penalty
    }
}
