#![allow(non_upper_case_globals)]

extern crate cuda;
extern crate float;

use ffi::*;

use cuda::runtime::*;
use float::stub::{f16_stub};

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

#[derive(Clone, Copy, Debug)]
pub enum CublasMathMode {
  Default,
  TensorOp,
}

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

pub trait CublasBlasExt<T> where T: Copy {
  unsafe fn gemv(&mut self,
      a_trans: CublasTranspose,
      m: i32, n: i32,
      alpha: *const T,
      a: *const T, lda: i32,
      x: *const T, incx: i32,
      beta: *const T,
      y: *mut T, incy: i32)
      -> CublasResult<()>;
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

unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

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

  pub fn set_math_mode(&mut self, math_mode: CublasMathMode) -> CublasResult<()> {
    let status = unsafe { cublasSetMathMode(self.as_mut_ptr(), math_mode.to_cublas()) };
    match status {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }
}

impl CublasBlasExt<f32> for CublasHandle {
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

impl CublasBlasExt<f64> for CublasHandle {
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

impl CublasBlasExt<f16_stub> for CublasHandle {
  unsafe fn gemv(&mut self,
      _a_trans: CublasTranspose,
      _m: i32, _n: i32,
      _alpha: *const f16_stub,
      _a: *const f16_stub, _lda: i32,
      _x: *const f16_stub, _incx: i32,
      _beta: *const f16_stub,
      _y: *mut f16_stub, _incy: i32)
      -> CublasResult<()>
  {
    // TODO: `cublasHgemv` does not exist.
    unimplemented!();
  }

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
    let status = cublasHgemm(
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
