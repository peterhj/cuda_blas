extern crate bindgen;

use std::env;
use std::path::{PathBuf};

fn main() {
  let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
  let cuda_dir = PathBuf::from(match env::var("CUDA_HOME") {
    Ok(path) => path,
    Err(_) => "/usr/local/cuda".to_owned(),
  });
  let cublas_bindings = bindgen::Builder::default()
    .clang_arg(format!("-I{}", cuda_dir.join("include").as_os_str().to_str().unwrap()))
    .header("wrap.h")
    .link("cublas")
    .whitelist_recursively(false)
    .whitelisted_type("cublasContext")
    .whitelisted_type("cublasHandle_t")
    .whitelisted_type("cublasStatus_t")
    .whitelisted_type("cublasOperation_t")
    .whitelisted_type("cublasPointerMode_t")
    .whitelisted_type("cublasAtomicsMode_t")
    //.whitelisted_type("cublasGemmAlgo_t")
    //.whitelisted_type("cudaDataType_t")
    // Helper functions.
    .whitelisted_function("cublasCreate_v2")
    .whitelisted_function("cublasDestroy_v2")
    .whitelisted_function("cublasSetStream_v2")
    .whitelisted_function("cublasSetPointerMode_v2")
    .whitelisted_function("cublasSetAtomicsMode")
    // Level 1 BLAS.
    .whitelisted_function("cublasSaxpy_v2")
    .whitelisted_function("cublasScopy_v2")
    .whitelisted_function("cublasSdot_v2")
    .whitelisted_function("cublasSnrm2_v2")
    .whitelisted_function("cublasSscal_v2")
    // Level 2 BLAS.
    .whitelisted_function("cublasSgemv_v2")
    // Level 3 BLAS.
    .whitelisted_function("cublasSgemm_v2")
    //.whitelisted_function("cublasSgemmBatched_v2")
    //.whitelisted_function("cublasSgemmStridedBatched_v2")
    // BLAS-like extensions.
    //.whitelisted_function("cublasGemmEx_v2")
    .generate()
    .expect("bindgen failed to generate cublas bindings");
  cublas_bindings
    .write_to_file(out_dir.join("cublas_bind.rs"))
    .expect("bindgen failed to write cublas bindings");
}
