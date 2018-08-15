#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use cuda::ffi::library_types::{cudaDataType};
use cuda::ffi::runtime::{cudaStream_t};

include!(concat!(env!("OUT_DIR"), "/cublas_bind.rs"));
