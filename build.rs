fn main() {
  println!("cargo:rustc-flags=-L /usr/local/cuda/lib64 -l dylib=cublas");
}
