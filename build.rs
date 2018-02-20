extern crate bindgen;

use std::env;
use std::path::{PathBuf};

fn main() {
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
  let cuda_dir = PathBuf::from(match env::var("CUDA_HOME") {
    Ok(path) => path,
    Err(_) => "/usr/local/cuda".to_owned(),
  });

  println!("cargo:rustc-link-lib=cublas");

  let cublas_bindings = bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrap.h")
    //.link("cublas")
    .whitelist_recursively(false)
    .whitelist_type("cublasContext")
    .whitelist_type("cublasHandle_t")
    .whitelist_type("cublasStatus_t")
    .whitelist_type("cublasOperation_t")
    .whitelist_type("cublasPointerMode_t")
    .whitelist_type("cublasAtomicsMode_t")
    //.whitelist_type("cublasGemmAlgo_t")
    //.whitelist_type("cudaDataType_t")
    // Helper functions.
    .whitelist_function("cublasCreate_v2")
    .whitelist_function("cublasDestroy_v2")
    .whitelist_function("cublasSetStream_v2")
    .whitelist_function("cublasSetPointerMode_v2")
    .whitelist_function("cublasSetAtomicsMode")
    // Level 1 BLAS.
    .whitelist_function("cublasSaxpy_v2")
    .whitelist_function("cublasScopy_v2")
    .whitelist_function("cublasSdot_v2")
    .whitelist_function("cublasSnrm2_v2")
    .whitelist_function("cublasSscal_v2")
    // Level 2 BLAS.
    .whitelist_function("cublasSgemv_v2")
    // Level 3 BLAS.
    .whitelist_function("cublasSgemm_v2")
    //.whitelist_function("cublasSgemmBatched_v2")
    //.whitelist_function("cublasSgemmStridedBatched_v2")
    // BLAS-like extensions.
    //.whitelist_function("cublasGemmEx_v2")
    .generate()
    .expect("bindgen failed to generate cublas bindings");
  cublas_bindings
    .write_to_file(out_dir.join("cublas_bind.rs"))
    .expect("bindgen failed to write cublas bindings");
}
