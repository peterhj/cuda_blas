#![feature(stmt_expr_attributes)]

fn main() {
  //println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l dylib=cublas");
  //println!("cargo:rustc-flags=-l dylib=cublas");

  println!("cargo:rustc-flags=-L /usr/local/cuda-7.0/lib64 -l dylib=cublas");

  /*#[cfg(feature = "cuda-7-0")]
  {
    println!("cargo:rustc-link-search=/usr/local/cuda-7.0/lib64");
  }

  #[cfg(feature = "cuda-7-5")]
  {
    println!("cargo:rustc-link-search=/usr/local/cuda-7.5/lib64");
  }*/
}
