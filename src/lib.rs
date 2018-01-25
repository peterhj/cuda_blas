#[allow(non_upper_case_globals)]

extern crate cuda;

use ffi::*;

use cuda::runtime::*;

//use std::os::raw::{c_int};
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
  pub fn to_bind_type(&self) -> cublasPointerMode_t {
    match self {
      &CublasPointerMode::Host   => cublasPointerMode_t_CUBLAS_POINTER_MODE_HOST,
      &CublasPointerMode::Device => cublasPointerMode_t_CUBLAS_POINTER_MODE_DEVICE,
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
  pub fn to_bind_type(&self) -> cublasOperation_t {
    match self {
      &CublasTranspose::N => cublasOperation_t_CUBLAS_OP_N,
      &CublasTranspose::T => cublasOperation_t_CUBLAS_OP_T,
      &CublasTranspose::H => cublasOperation_t_CUBLAS_OP_C,
    }
  }
}

pub trait CublasBlasExt<T> where T: Copy {
  unsafe fn gemv(&self,
      a_trans: CublasTranspose,
      m: i32, n: i32,
      alpha: *const T,
      a: *const T, lda: i32,
      x: *const T, incx: i32,
      beta: *const T,
      y: *mut T, incy: i32)
      -> CublasResult<()>;
  unsafe fn gemm(&self,
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

impl CublasHandle {
  pub fn create() -> CublasResult<CublasHandle> {
    let mut handle: cublasHandle_t = 0 as cublasHandle_t;
    let status_code = unsafe { cublasCreate_v2(&mut handle as *mut cublasHandle_t) };
    match status_code {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(CublasHandle{ptr: handle}),
      e => Err(CublasError(e)),
    }
  }

  pub fn reset_stream(&self) -> CublasResult<()> {
    let status_code = unsafe { cublasSetStream_v2(self.as_ptr(), null_mut()) };
    match status_code {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }

  pub fn set_stream(&self, stream: &CudaStream) -> CublasResult<()> {
    let status_code = unsafe { cublasSetStream_v2(self.as_ptr(), stream.as_ptr()) };
    match status_code {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }

  pub fn set_pointer_mode(&self, pointer_mode: CublasPointerMode) -> CublasResult<()> {
    let status_code = unsafe { cublasSetPointerMode_v2(self.as_ptr(), pointer_mode.to_bind_type()) };
    match status_code {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      e => Err(CublasError(e)),
    }
  }

  pub unsafe fn as_ptr(&self) -> cublasHandle_t {
    self.ptr
  }
}

impl CublasBlasExt<f32> for CublasHandle {
  unsafe fn gemv(&self,
      a_trans: CublasTranspose,
      m: i32, n: i32,
      alpha: *const f32,
      a: *const f32, lda: i32,
      x: *const f32, incx: i32,
      beta: *const f32,
      y: *mut f32, incy: i32)
      -> CublasResult<()>
  {
    let e = cublasSgemv_v2(
        self.as_ptr(),
        a_trans.to_bind_type(),
        m, n,
        alpha,
        a, lda,
        x, incx,
        beta,
        y, incy);
    match e {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(e)),
    }
  }

  unsafe fn gemm(&self,
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
    let e = cublasSgemm_v2(
        self.as_ptr(),
        a_trans.to_bind_type(),
        b_trans.to_bind_type(),
        m, n, k,
        alpha,
        a, lda,
        b, ldb,
        beta,
        c, ldc);
    match e {
      cublasStatus_t_CUBLAS_STATUS_SUCCESS => Ok(()),
      _ => Err(CublasError(e)),
    }
  }
}
