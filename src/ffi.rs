#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use cuda::ffi::runtime::{cudaStream_t};
use float::stub::{f16_stub};
use std::os::raw::{c_int};

include!(concat!(env!("OUT_DIR"), "/cublas_bind.rs"));

extern "C" {
  // NOTE: Manually bind this, as it is inside an `#ifdef __cplusplus`
  // and bindgen chokes when parsing "cublas_v2.h" in C++ mode.
  pub fn cublasHgemm(
      handle: cublasHandle_t,
      transa: cublasOperation_t,
      transb: cublasOperation_t,
      m: c_int,
      n: c_int,
      k: c_int,
      alpha: *const f16_stub,
      A: *const f16_stub,
      lda: c_int,
      B: *const f16_stub,
      ldb: c_int,
      beta: *const f16_stub,
      C: *mut f16_stub,
      ldc: c_int) -> cublasStatus_t;
}
