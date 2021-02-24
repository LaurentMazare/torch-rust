// The LIBTORCH environment variable can be used to specify the directory
// where libtorch has been installed.
// When not specified this script downloads the cpu version for libtorch
// and extracts it in OUT_DIR.
//
// On Linux, the TORCH_CUDA_VERSION environment variable can be used,
// like 9.0, 90, or cu90 to specify the version of CUDA to use for libtorch.

use anyhow::{bail, Result};
use cmake::Config;
use curl::easy::Easy;
use serde::{de::Error as _, Deserialize, Deserializer};
use std::{
    fs, io,
    io::Write,
    path::{Path, PathBuf},
};

use env_config::*;

const TORCH_VERSION: &str = "1.7.0";
const RETURN_ENV_VARS: [&str; 4] = [
    "TORCH_CUDA_VERSION",
    "LIBTORCH",
    "LIBTORCH_CXX11_ABI",
    "LIBTORCH_USE_CMAKE",
];

mod env_config {
    use super::*;

    #[derive(Debug, Clone, Deserialize)]
    pub struct EnvConfig {
        #[serde(default = "default_torch_cuda_version")]
        pub torch_cuda_version: CudaVersion,
        pub libtorch: Option<PathBuf>,
        #[serde(
            deserialize_with = "deserialize_bool",
            default = "default_libtorch_cxx11_abi"
        )]
        pub libtorch_cxx11_abi: bool,
        #[serde(
            deserialize_with = "deserialize_bool",
            default = "default_libtorch_use_cmake"
        )]
        pub libtorch_use_cmake: bool,
        pub cargo_cfg_target_os: Os,
        pub out_dir: PathBuf,
        pub target: String,
    }

    #[derive(Debug, Clone, Copy, Deserialize)]
    pub enum CudaVersion {
        #[serde(rename = "cpu")]
        Cpu,
        #[serde(rename = "cu92")]
        Cu92,
        #[serde(rename = "cu101")]
        Cu101,
        #[serde(rename = "cu110")]
        Cu110,
    }

    impl CudaVersion {
        pub fn as_str(&self) -> &str {
            match self {
                Self::Cpu => "cpu",
                Self::Cu92 => "cpu92",
                Self::Cu101 => "cpu101",
                Self::Cu110 => "cpu110",
            }
        }
    }

    #[derive(Debug, Clone)]
    pub enum Os {
        Linux,
        MacOs,
        Windows,
        Other(String),
    }

    impl<'de> Deserialize<'de> for Os {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            let text = String::deserialize(deserializer)?;

            let os = match text.as_str() {
                "linux" => Os::Linux,
                "macos" => Os::MacOs,
                "windows" => Os::Windows,
                _ => Os::Other(text),
            };
            Ok(os)
        }
    }

    fn deserialize_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
    where
        D: Deserializer<'de>,
    {
        let text = String::deserialize(deserializer)?;
        let value = match text.as_str() {
            "1" => true,
            "0" => false,
            _ => {
                return Err(D::Error::custom(format!(
                    "expect 0 or 1, but get '{}'",
                    text
                )))
            }
        };
        Ok(value)
    }

    fn default_torch_cuda_version() -> CudaVersion {
        CudaVersion::Cpu
    }

    fn default_libtorch_cxx11_abi() -> bool {
        true
    }

    fn default_libtorch_use_cmake() -> bool {
        false
    }
}

fn main() -> Result<()> {
    // load environment variables
    let config: EnvConfig = envy::from_env()?;
    eprintln!("=== environment variables ===");
    eprintln!("{:#?}", config);

    // skip build if doc only mode
    if cfg!(feature = "doc-only") {
        return Ok(());
    }

    // prepare libtorch directory
    let libtorch = prepare_libtorch_dir(&config)?;

    // detect cuda and hip
    // use_cuda is a hacky way to detect whether cuda is available and
    // if it's the case link to it by explicitly depending on a symbol
    // from the torch_cuda library.
    // It would be better to use -Wl,--no-as-needed but there is no way
    // to specify arbitrary linker flags at the moment.
    //
    // Once https://github.com/rust-lang/cargo/pull/8441 is available
    // we should switch to using rustc-link-arg instead e.g. with the
    // following flags:
    //   -Wl,--no-as-needed -Wl,--copy-dt-needed-entries -ltorch
    //
    // This will be available starting from cargo 1.50 but will be a nightly
    // only option to start with.
    // https://github.com/rust-lang/cargo/blob/master/CHANGELOG.md
    let use_cuda = detect_cuda(&config, &libtorch);
    let use_hip = detect_hip(&config, &libtorch);

    // build ffi library
    if config.libtorch_use_cmake {
        cmake(&libtorch)?;
    } else {
        make(&config, &libtorch, use_cuda, use_hip)?;
    }

    // add environment variables re-run guards
    RETURN_ENV_VARS.iter().for_each(|name| {
        println!("cargo:rerun-if-env-changed={}", name);
    });

    // add link search path
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch.join("lib").display()
    );

    // link libraries
    println!("cargo:rustc-link-lib=static=tch");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=c10");
    if use_cuda {
        println!("cargo:rustc-link-lib=torch_cuda");
    }
    if use_hip {
        println!("cargo:rustc-link-lib=torch_hip");
        println!("cargo:rustc-link-lib=c10_hip");
    }
    if !config.target.contains("msvc") && !config.target.contains("apple") {
        println!("cargo:rustc-link-lib=gomp");
    }

    Ok(())
}

fn download(source_url: &str, target_file: impl AsRef<Path>) -> Result<()> {
    let f = fs::File::create(&target_file)?;
    let mut writer = io::BufWriter::new(f);
    let mut easy = Easy::new();
    easy.url(&source_url)?;
    easy.write_function(move |data| Ok(writer.write(data).unwrap()))?;
    easy.perform()?;
    let response_code = easy.response_code()?;
    if response_code == 200 {
        Ok(())
    } else {
        bail!(
            "Unexpected response code {} for {}",
            response_code,
            source_url
        )
    }
}

fn extract(filename: impl AsRef<Path>, outpath: impl AsRef<Path>) -> Result<()> {
    let file = fs::File::open(&filename)?;
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        #[allow(deprecated)]
        let outpath = outpath.as_ref().join(file.sanitized_name());
        if !(&*file.name()).ends_with('/') {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.as_path().display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(&p)?;
                }
            }
            let mut outfile = fs::File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(())
}

fn prepare_libtorch_dir(config: &EnvConfig) -> Result<PathBuf> {
    let libtorch_dir = match &config.libtorch {
        Some(path) => path.to_owned(),
        None => {
            let libtorch_dir = config.out_dir.join("libtorch");
            if !libtorch_dir.exists() {
                fs::create_dir(&libtorch_dir).unwrap_or_default();
                let device = config.torch_cuda_version.as_str();
                let libtorch_url = match config.cargo_cfg_target_os {
                    Os::Linux => format!(
                        "https://download.pytorch.org/libtorch/{}/libtorch-cxx11-abi-shared-with-deps-{}%2B{}.zip",
                        device, TORCH_VERSION, device
                    ),
                    Os::MacOs => format!(
                        "https://download.pytorch.org/libtorch/cpu/libtorch-macos-{}.zip",
                        TORCH_VERSION
                    ),
                    Os::Windows => format!(
                        "https://download.pytorch.org/libtorch/{}/libtorch-win-shared-with-deps-{}%2B{}.zip",
                        device, TORCH_VERSION, device
                    ),
                    _ => bail!("Unsupported OS"),
                };

                let filename = libtorch_dir.join(format!("v{}.zip", TORCH_VERSION));
                download(&libtorch_url, &filename)?;
                extract(&filename, &libtorch_dir)?;
            }

            libtorch_dir.join("libtorch")
        }
    };

    Ok(libtorch_dir)
}

fn make(
    config: &EnvConfig,
    libtorch: impl AsRef<Path>,
    use_cuda: bool,
    use_hip: bool,
) -> Result<()> {
    let libtorch = libtorch.as_ref();
    let cuda_dependency = if use_cuda || use_hip {
        "libtch/dummy_cuda_dependency.cpp"
    } else {
        "libtch/fake_cuda_dependency.cpp"
    };
    println!("cargo:rerun-if-changed=libtch/torch_api.cpp");
    println!("cargo:rerun-if-changed=libtch/torch_api.h");
    println!("cargo:rerun-if-changed=libtch/torch_api_generated.cpp.h");
    println!("cargo:rerun-if-changed=libtch/torch_api_generated.h");
    println!("cargo:rerun-if-changed=libtch/stb_image_write.h");
    println!("cargo:rerun-if-changed=libtch/stb_image_resize.h");
    println!("cargo:rerun-if-changed=libtch/stb_image.h");
    match &config.cargo_cfg_target_os {
        Os::Linux | Os::MacOs => {
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch.join("include"))
                .include(libtorch.join("include/torch/csrc/api/include"))
                .flag(&format!("-Wl,-rpath={}", libtorch.join("lib").display()))
                .flag("-std=c++14")
                .flag(&format!(
                    "-D_GLIBCXX_USE_CXX11_ABI={}",
                    if config.libtorch_cxx11_abi { "1" } else { "0" }
                ))
                .file("libtch/torch_api.cpp")
                .file(cuda_dependency)
                .compile("tch");
        }
        Os::Windows => {
            // TODO: Pass "/link" "LIBPATH:{}" to cl.exe in order to emulate rpath.
            //       Not yet supported by cc=rs.
            //       https://github.com/alexcrichton/cc-rs/issues/323
            cc::Build::new()
                .cpp(true)
                .pic(true)
                .warnings(false)
                .include(libtorch.join("include"))
                .include(libtorch.join("include/torch/csrc/api/include"))
                .file("libtch/torch_api.cpp")
                .file(cuda_dependency)
                .compile("tch");
        }
        Os::Other(name) => bail!("Unsupported OS {}", name),
    };

    Ok(())
}

fn cmake(libtorch: impl AsRef<Path>) -> Result<()> {
    let dst = Config::new("libtch")
        .define("CMAKE_PREFIX_PATH", libtorch.as_ref())
        .build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=tch");
    println!("cargo:rustc-link-lib=stdc++");

    Ok(())
}

fn detect_cuda(config: &EnvConfig, libtorch: impl AsRef<Path>) -> bool {
    let libtorch = libtorch.as_ref();

    match config.cargo_cfg_target_os {
        Os::Linux => libtorch.join("lib").join("libtorch_cuda.so").exists(),
        Os::Windows => libtorch.join("lib").join("torch_cuda.dll").exists(),
        Os::MacOs | Os::Other(_) => false,
    }
}

fn detect_hip(config: &EnvConfig, libtorch: impl AsRef<Path>) -> bool {
    let libtorch = libtorch.as_ref();

    match config.cargo_cfg_target_os {
        Os::Linux => libtorch.join("lib").join("libtorch_hip.so").exists(),
        Os::Windows => libtorch.join("lib").join("torch_hip.dll").exists(),
        Os::MacOs | Os::Other(_) => false,
    }
}
