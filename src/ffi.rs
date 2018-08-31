#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use cuda::ffi::driver_types::{cudaStream_t};
use cuda::ffi::library_types::{cudaDataType};

include!(concat!(env!("OUT_DIR"), "/cublas_bind.rs"));
