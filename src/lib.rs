#![allow(non_upper_case_globals)]

extern crate cuda;
#[cfg(feature = "f16")] extern crate float;

use ffi::*;

use cuda::ffi::library_types::*;
use cuda::runtime::*;
#[cfg(feature = "f16")] use float::stub::{f16_stub};

use std::ptr::{null_mut};

pub mod ffi;

#[derive(Clone, Copy, Debug)]
pub struct CublasError(pub cublasStatus_t);

pub type CublasResult<T> = Result<T, CublasError>;

#[derive(Clone, Copy, Debug)]
pub enum CublasPointerMode {
  Host,
  Device,
}

impl CublasPointerMode {
  pub fn to_cublas(&self) -> cublasPointerMode_t {
    match self {
      &CublasPointerMode::Host   => cublasPointerMode_t_CUBLAS_POINTER_MODE_HOST,
      &CublasPointerMode::Device => cublasPointerMode_t_CUBLAS_POINTER_MODE_DEVICE,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum CublasAtomicsMode {
  NotAllowed,
  Allowed,
}

impl CublasAtomicsMode {
  pub fn to_cublas(&self) -> cublasAtomicsMode_t {
    match self {
      &CublasAtomicsMode::NotAllowed    => cublasAtomicsMode_t_CUBLAS_ATOMICS_NOT_ALLOWED,
      &CublasAtomicsMode::Allowed       => cublasAtomicsMode_t_CUBLAS_ATOMICS_ALLOWED,
    }
  }
}

#[cfg(feature = "cuda9")]
#[derive(Clone, Copy, Debug)]
pub enum CublasMathMode {
  Default,
  TensorOp,
}

#[cfg(feature = "cuda9")]
impl CublasMathMode {
  pub fn to_cublas(&self) -> cublasMath_t {
    match self {
      &CublasMathMode::Default  => cublasMath_t_CUBLAS_DEFAULT_MATH,
      &CublasMathMode::TensorOp => cublasMath_t_CUBLAS_TENSOR_OP_MATH,
    }
  }
}

#[derive(Clone, Copy, Debug)]
pub enum CublasTranspose {
  N,
  T,
  H,
}

impl CublasTranspose {
  pub fn to_cublas(&self) -> cublasOperation_t {
    match self {
      &CublasTranspose::N => cublasOperation_t_CUBLAS_OP_N,
      &CublasTranspose::T => cublasOperation_t_CUBLAS_OP_T,
      &CublasTranspose::H => cublasOperation_t_CUBLAS_OP_C,
    }
  }
}

pub trait CublasBlas1Ext<T> where T: Copy {
  unsafe fn nrm2(&mut self,
      n: i32,
      x: *const T, incx: i32,
      result: *mut T)
      -> CublasResult<()>;
}

pub trait CublasBlas2Ext<T> where T: Copy {
  unsafe fn gemv(&mut self,
      a_trans: CublasTranspose,
      m: i32, n: i32,
      alpha: *const T,
      a: *const T, lda: i32,
      x: *const T, incx: i32,
      beta: *const T,
      y: *mut T, incy: i32)
      -> CublasResult<()>;
}

pub trait CublasBlas3Ext<T> where T: Copy {
  unsafe fn gemm(&mut self,
      a_trans: CublasTranspose,
      b_trans: CublasTranspose,
      m: i32, n: i32, k: i32,
      alpha: *const T,
      a: *const T, lda: i32,
      b: *const T, ldb: i32,
      beta: *const T,
      c: *mut T, ldc: i32)
      -> CublasResult<()>;
}

pub struct CublasHandle {
  ptr: cublasHandle_t,
}

impl Drop for CublasHandle {
  fn drop(&mut self) {
    let status = unsafe { cublasDestroy_v2(self.ptr) };
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => {}
      e => panic!("{}", e),
    }
  }
}

impl CublasHandle {
  pub fn create() -> CublasResult<CublasHandle> {
    let mut ptr = null_mut();
    let status = unsafe { cublasCreate_v2(&mut ptr as *mut cublasHandle_t) };
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(CublasHandle{ptr: ptr}),
      e => Err(CublasError(e)),
    }
  }

  pub unsafe fn as_mut_ptr(&mut self) -> cublasHandle_t {
    self.ptr
  }

  pub fn get_version(&mut self) -> CublasResult<i32> {
    let mut version: i32 = -1;
    let status = unsafe { cublasGetVersion_v2(self.as_mut_ptr(), &mut version as *mut _) };
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(version),
      e => Err(CublasError(e)),
    }
  }

  pub fn set_stream(&mut self, stream: &mut CudaStream) -> CublasResult<()> {
    let status = unsafe { cublasSetStream_v2(self.as_mut_ptr(), stream.as_mut_ptr()) };
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }

  pub fn set_pointer_mode(&mut self, pointer_mode: CublasPointerMode) -> CublasResult<()> {
    let status = unsafe { cublasSetPointerMode_v2(self.as_mut_ptr(), pointer_mode.to_cublas()) };
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }

  pub fn set_atomics_mode(&mut self, atomics_mode: CublasAtomicsMode) -> CublasResult<()> {
    let status = unsafe { cublasSetAtomicsMode(self.as_mut_ptr(), atomics_mode.to_cublas()) };
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }

  #[cfg(feature = "cuda9")]
  pub fn set_math_mode(&mut self, math_mode: CublasMathMode) -> CublasResult<()> {
    let status = unsafe { cublasSetMathMode(self.as_mut_ptr(), math_mode.to_cublas()) };
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }
}

impl CublasBlas1Ext<f32> for CublasHandle {
  unsafe fn nrm2(&mut self,
      n: i32,
      x: *const f32, incx: i32,
      result: *mut f32)
      -> CublasResult<()>
  {
    let status = cublasSnrm2_v2(
        self.as_mut_ptr(),
        n,
        x, incx,
        result);
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }
}

impl CublasBlas2Ext<f32> for CublasHandle {
  unsafe fn gemv(&mut self,
      a_trans: CublasTranspose,
      m: i32, n: i32,
      alpha: *const f32,
      a: *const f32, lda: i32,
      x: *const f32, incx: i32,
      beta: *const f32,
      y: *mut f32, incy: i32)
      -> CublasResult<()>
  {
    let status = cublasSgemv_v2(
        self.as_mut_ptr(),
        a_trans.to_cublas(),
        m, n,
        alpha,
        a, lda,
        x, incx,
        beta,
        y, incy);
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }
}

impl CublasBlas3Ext<f32> for CublasHandle {
  unsafe fn gemm(&mut self,
      a_trans: CublasTranspose,
      b_trans: CublasTranspose,
      m: i32, n: i32, k: i32,
      alpha: *const f32,
      a: *const f32, lda: i32,
      b: *const f32, ldb: i32,
      beta: *const f32,
      c: *mut f32, ldc: i32)
      -> CublasResult<()>
  {
    let status = cublasSgemm_v2(
        self.as_mut_ptr(),
        a_trans.to_cublas(),
        b_trans.to_cublas(),
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }
}

impl CublasBlas1Ext<f64> for CublasHandle {
  unsafe fn nrm2(&mut self,
      n: i32,
      x: *const f64, incx: i32,
      result: *mut f64)
      -> CublasResult<()>
  {
    let status = cublasDnrm2_v2(
        self.as_mut_ptr(),
        n,
        x, incx,
        result);
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }
}

impl CublasBlas2Ext<f64> for CublasHandle {
  unsafe fn gemv(&mut self,
      a_trans: CublasTranspose,
      m: i32, n: i32,
      alpha: *const f64,
      a: *const f64, lda: i32,
      x: *const f64, incx: i32,
      beta: *const f64,
      y: *mut f64, incy: i32)
      -> CublasResult<()>
  {
    let status = cublasDgemv_v2(
        self.as_mut_ptr(),
        a_trans.to_cublas(),
        m, n,
        alpha,
        a, lda,
        x, incx,
        beta,
        y, incy);
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }
}

impl CublasBlas3Ext<f64> for CublasHandle {
  unsafe fn gemm(&mut self,
      a_trans: CublasTranspose,
      b_trans: CublasTranspose,
      m: i32, n: i32, k: i32,
      alpha: *const f64,
      a: *const f64, lda: i32,
      b: *const f64, ldb: i32,
      beta: *const f64,
      c: *mut f64, ldc: i32)
      -> CublasResult<()>
  {
    let status = cublasDgemm_v2(
        self.as_mut_ptr(),
        a_trans.to_cublas(),
        b_trans.to_cublas(),
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }
}

#[cfg(feature = "f16")]
impl CublasBlas3Ext<f16_stub> for CublasHandle {
  unsafe fn gemm(&mut self,
      a_trans: CublasTranspose,
      b_trans: CublasTranspose,
      m: i32, n: i32, k: i32,
      alpha: *const f16_stub,
      a: *const f16_stub, lda: i32,
      b: *const f16_stub, ldb: i32,
      beta: *const f16_stub,
      c: *mut f16_stub, ldc: i32)
      -> CublasResult<()>
  {
    let status = cublasGemmEx(
        self.as_mut_ptr(),
        a_trans.to_cublas(),
        b_trans.to_cublas(),
        m, n, k,
        alpha as *const _,
        a as *const _, cudaDataType_t_CUDA_R_16F, lda,
        b as *const _, cudaDataType_t_CUDA_R_16F, ldb,
        beta as *const _,
        c as *mut _, cudaDataType_t_CUDA_R_16F, ldc,
        cudaDataType_t_CUDA_R_16F,
        cublasGemmAlgo_t_CUBLAS_GEMM_DFALT_TENSOR_OP);
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }
}
