#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use cuda::ffi::runtime::{cudaStream_t};
use float::stub::{f16_stub};

pub type __half = f16_stub;

include!(concat!(env!("OUT_DIR"), "/cublas_bind.rs"));
