#[cfg(feature = "generate_model_apis")]
#[macro_use]
extern crate bart_derive;

use std::env;
use std::path::{Path, PathBuf};

fn manifest_dir() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
}

fn submodules() -> PathBuf {
    manifest_dir().join("submodules")
}

fn binary_changing_features() -> String {
    let mut features = String::new();
    if cfg!(feature = "debug_tflite") {
        features.push_str("-debug");
    }
    if cfg!(feature = "no_micro") {
        features.push_str("-no_micro");
    }
    features
}

fn prepare_tensorflow_library() {
    let _arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");

    #[cfg(feature = "build")]
    {
        let out_dir = env::var("OUT_DIR").unwrap();
        let submodules = submodules();
        let tf_src_dir = submodules.join("tensorflow/tensorflow/lite");
        let cmake_build_dir = Path::new(&out_dir).join("tflite_cmake_build");
        let cmake_build_dir_str = cmake_build_dir.to_string_lossy();
        let tf_lib_name = cmake_build_dir.join("libtensorflow-lite.a");
        let binary_changing_features = binary_changing_features();

        let target = env::var("TARGET").unwrap_or_else(|_| "native".to_string());
        let is_cross_compile =
            target != "native" && target != env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

        if !tf_lib_name.exists() {
            std::fs::create_dir_all(&cmake_build_dir).expect("Unable to create cmake build dir");

            let mut cmake_config = std::process::Command::new("cmake");
            cmake_config.arg(tf_src_dir.to_string_lossy().to_string());
            cmake_config.arg("-DCMAKE_BUILD_TYPE=Release");
            cmake_config.arg("-DCMAKE_POLICY_VERSION_MINIMUM=3.5");
            cmake_config.arg("-DFLATBUFFERS_BUILD_FLATC=OFF");
            cmake_config.arg("-DFLATBUFFERS_BUILD_FLATHASH=OFF");
            cmake_config.arg("-DFLATBUFFERS_BUILD_GRPC=OFF");
            cmake_config.arg("-DFLATBUFFERS_INSTALL=OFF");
            cmake_config.arg("-DFLATBUFFERS_BUILD_TESTS=OFF");
            cmake_config.arg("-DBUILD_SHARED_LIBS=ON");

            if is_cross_compile {
                let toolchain_file = match target.as_str() {
                    "aarch64" => "${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-aarch64-linux-gnu/bin/aarch64-linux-gnu-",
                    "armv7" => "${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-arm-linux-gnueabihf/bin/arm-linux-gnueabihf-",
                    "aarch64-apple-darwin" => "${HOME}/toolchains/gcc-arm-8.3-2019.03-x86_64-aarch64-apple-darwin/bin/aarch64-apple-darwin-",
                    _ => panic!("Unsupported target architecture: {target}"),
                };

                cmake_config.arg("-DCMAKE_TOOLCHAIN_FILE=".to_string() + toolchain_file);
                cmake_config.arg(format!("-DCMAKE_SYSTEM_PROCESSOR={target}"));
            }

            cmake_config.current_dir(&cmake_build_dir);
            assert!(
                cmake_config.status().expect("Failed to run cmake configure").success(),
                "CMake configuration failed"
            );

            let mut cmake_build = std::process::Command::new("cmake");
            cmake_build.arg("--build");
            cmake_build.arg(".");
            let num_jobs =
                std::env::var("NUM_JOBS").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(8);
            cmake_build.arg(format!("-j{num_jobs}"));
            cmake_build.current_dir(&cmake_build_dir);
            assert!(
                cmake_build.status().expect("Failed to run cmake build").success(),
                "CMake build failed"
            );
        }

        println!("cargo:rustc-link-search=native={cmake_build_dir_str}");
        println!("cargo:rustc-link-lib=static=tensorflow-lite{binary_changing_features}");
    }
    #[cfg(not(feature = "build"))]
    {
        let arch_var = format!("TFLITE_{}_LIB_DIR", arch.replace("-", "_").to_uppercase());
        let all_var = "TFLITE_LIB_DIR".to_string();
        let lib_dir = env::var(&arch_var).or(env::var(&all_var)).unwrap_or_else(|_| {
            panic!(
                "[feature = build] not set and environment variables {} and {} are not set",
                arch_var, all_var
            )
        });
        println!("cargo:rustc-link-search=native={}", lib_dir);
        let static_dynamic = if Path::new(&lib_dir).join("libtensorflow-lite.a").exists() {
            "static"
        } else {
            "dylib"
        };
        println!("cargo:rustc-link-lib={}=tensorflow-lite", static_dynamic);
        println!("cargo:rerun-if-changed={}", lib_dir);
    }
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
}

// This generates "tflite_types.rs" containing structs and enums which are inter-operable with Glow.
fn import_tflite_types() {
    use bindgen::{Builder, CodegenConfig, EnumVariation};

    let submodules = submodules();
    let submodules_str = submodules.to_string_lossy();
    let bindings = Builder::default()
        .allowlist_recursively(true)
        .prepend_enum_name(false)
        .impl_debug(true)
        .with_codegen_config(CodegenConfig::TYPES)
        .layout_tests(false)
        .enable_cxx_namespaces()
        .derive_default(true)
        .size_t_is_usize(true)
        // for model APIs
        .allowlist_type("tflite::ModelT")
        .allowlist_type(".+OptionsT")
        .blocklist_type(".+_TableType")
        // for interpreter
        .allowlist_type("tflite::FlatBufferModel")
        .opaque_type("tflite::FlatBufferModel")
        .allowlist_type("tflite::InterpreterBuilder")
        .opaque_type("tflite::InterpreterBuilder")
        .allowlist_type("tflite::Interpreter")
        .opaque_type("tflite::Interpreter")
        .allowlist_type("tflite::ops::builtin::BuiltinOpResolver")
        .opaque_type("tflite::ops::builtin::BuiltinOpResolver")
        .allowlist_type("tflite::OpResolver")
        .opaque_type("tflite::OpResolver")
        .allowlist_type("TfLiteTensor")
        .opaque_type("std::string")
        .opaque_type("std::basic_string.*")
        .opaque_type("std::map.*")
        .opaque_type("flatbuffers::NativeTable")
        .blocklist_type("std")
        .blocklist_type("tflite::Interpreter_TfLiteDelegatePtr")
        .blocklist_type("tflite::Interpreter_State")
        .default_enum_style(EnumVariation::Rust { non_exhaustive: false })
        .derive_partialeq(true)
        .derive_eq(true)
        .header("csrc/tflite_wrapper.hpp")
        .clang_arg(format!("-I{submodules_str}/tensorflow"))
        .clang_arg(format!("-I{submodules_str}/flatbuffers/include"))
        .clang_arg("-DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK")
        .clang_arg("-DFLATBUFFERS_POLYMORPHIC_NATIVETABLE")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        // required to get cross compilation for aarch64 to work because of an issue in flatbuffers
        .clang_arg("-fms-extensions")
        .no_copy("_Tp");

    let bindings = bindings.generate().expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/tflite_types.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("tflite_types.rs");
    let bindings = bindings
        .to_string()
        .replace("pub _M_val: _Tp", "pub _M_val: std::mem::ManuallyDrop<_Tp>")
        .replace("Vector_iterator", "Vector_iterator<Data>")
        .replace("Vector_reverse_iterator", "Vector_reverse_iterator<Data>")
        .replace(
            "type _Rb_tree_insert_return_type",
            "type _Rb_tree_insert_return_type<_Iterator,_NodeHandle>",
        );
    std::fs::write(out_path, bindings).expect("Couldn't write bindings!");
}

fn build_inline_cpp() {
    let submodules = submodules();

    cpp_build::Config::new()
        .include(submodules.join("tensorflow"))
        .include(submodules.join("flatbuffers/include"))
        .flag("-fPIC")
        .flag("-std=c++17")
        .flag("-Wno-sign-compare")
        .define("GEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK", None)
        .define("FLATBUFFERS_POLYMORPHIC_NATIVETABLE", None)
        .debug(true)
        .opt_level(if cfg!(debug_assertions) { 0 } else { 2 })
        .build("src/lib.rs");
}

fn import_stl_types() {
    use bindgen::*;

    let bindings = Builder::default()
        .enable_cxx_namespaces()
        .allowlist_type("std::string")
        .opaque_type("std::string")
        .allowlist_type("rust::.+")
        .opaque_type("rust::.+")
        .blocklist_type("std")
        .header("csrc/stl_wrapper.hpp")
        .layout_tests(false)
        .derive_partialeq(true)
        .derive_eq(true)
        .clang_arg("-include")
        .clang_arg("cstdint")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg("-fms-extensions")
        .formatter(bindgen::Formatter::Rustfmt)
        .generate()
        .expect("Unable to generate STL bindings");

    // Write the bindings to the $OUT_DIR/tflite_types.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join("stl_types.rs");
    bindings.write_to_file(out_path).expect("Couldn't write bindings!");
}

#[cfg(feature = "generate_model_apis")]
fn generate_memory_impl() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut file = std::fs::File::create("src/model/stl/memory_impl.rs")?;
    writeln!(
        &mut file,
        r#"
#![allow(clippy::transmute_num_to_bytes)]
use std::{{fmt, mem}};
use std::ops::{{Deref, DerefMut}};

use crate::model::stl::memory::UniquePtr;
"#
    )?;

    #[derive(BartDisplay)]
    #[template = "data/memory_basic_impl.rs.template"]
    struct MemoryBasicImpl<'a> {
        cpp_type: &'a str,
        rust_type: &'a str,
    }

    let memory_types = vec![
        ("OperatorCodeT", "crate::model::OperatorCodeT"),
        ("TensorT", "crate::model::TensorT"),
        ("OperatorT", "crate::model::OperatorT"),
        ("SubGraphT", "crate::model::SubGraphT"),
        ("BufferT", "crate::model::BufferT"),
        ("QuantizationParametersT", "crate::model::QuantizationParametersT"),
        ("ModelT", "crate::model::ModelT"),
        ("MetadataT", "crate::model::MetadataT"),
        ("TensorMapT", "crate::model::TensorMapT"),
        ("SignatureDefT", "crate::model::SignatureDefT"),
    ];

    for (cpp_type, rust_type) in memory_types {
        writeln!(&mut file, "{}\n", &MemoryBasicImpl { cpp_type, rust_type },)?;
    }
    Ok(())
}

#[cfg(feature = "generate_model_apis")]
fn generate_vector_impl() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut file = std::fs::File::create("src/model/stl/vector_impl.rs")?;
    writeln!(
        &mut file,
        r#"
#![allow(clippy::transmute_num_to_bytes)]
use std::{{fmt, mem, slice}};
use std::ops::{{Deref, DerefMut, Index, IndexMut}};

use libc::size_t;

use super::memory::UniquePtr;
use super::vector::{{VectorOfUniquePtr, VectorErase, VectorExtract, VectorInsert, VectorSlice}};
use crate::model::stl::bindings::root::rust::dummy_vector;

cpp! {{{{
    #include <vector>
}}}}
"#
    )?;

    #[derive(BartDisplay)]
    #[template = "data/vector_primitive_impl.rs.template"]
    #[allow(non_snake_case)]
    struct VectorPrimitiveImpl<'a> {
        cpp_type: &'a str,
        rust_type: &'a str,
        RustType: &'a str,
    }

    let vector_types = vec![
        ("uint8_t", "u8", "U8"),
        ("int32_t", "i32", "I32"),
        ("int64_t", "i64", "I64"),
        ("float", "f32", "F32"),
    ];

    #[allow(non_snake_case)]
    for (cpp_type, rust_type, RustType) in vector_types {
        writeln!(&mut file, "{}\n", &VectorPrimitiveImpl { cpp_type, rust_type, RustType },)?;
    }

    #[derive(BartDisplay)]
    #[template = "data/vector_basic_impl.rs.template"]
    struct VectorBasicImpl<'a> {
        cpp_type: &'a str,
        rust_type: &'a str,
    }

    let vector_types = vec![
        ("std::unique_ptr<OperatorCodeT>", "UniquePtr<crate::model::OperatorCodeT>"),
        ("std::unique_ptr<TensorT>", "UniquePtr<crate::model::TensorT>"),
        ("std::unique_ptr<OperatorT>", "UniquePtr<crate::model::OperatorT>"),
        ("std::unique_ptr<SubGraphT>", "UniquePtr<crate::model::SubGraphT>"),
        ("std::unique_ptr<BufferT>", "UniquePtr<crate::model::BufferT>"),
        ("std::unique_ptr<MetadataT>", "UniquePtr<crate::model::MetadataT>"),
        ("std::unique_ptr<SignatureDefT>", "UniquePtr<crate::model::SignatureDefT>"),
        ("std::unique_ptr<TensorMapT>", "UniquePtr<crate::model::TensorMapT>"),
    ];

    for (cpp_type, rust_type) in vector_types {
        writeln!(&mut file, "{}\n", &VectorBasicImpl { cpp_type, rust_type },)?;
    }
    Ok(())
}

#[cfg(feature = "generate_model_apis")]
fn generate_builtin_options_impl() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut file = std::fs::File::create("src/model/builtin_options_impl.rs")?;
    writeln!(
        &mut file,
        r#"
use super::{{BuiltinOptions, BuiltinOptionsUnion, NativeTable}};
"#
    )?;

    #[derive(BartDisplay)]
    #[template = "data/builtin_options_impl.rs.template"]
    struct BuiltinOptionsImpl<'a> {
        name: &'a str,
    }

    let option_names = vec![
        "Conv2DOptions",
        "DepthwiseConv2DOptions",
        "ConcatEmbeddingsOptions",
        "LSHProjectionOptions",
        "Pool2DOptions",
        "SVDFOptions",
        "RNNOptions",
        "FullyConnectedOptions",
        "SoftmaxOptions",
        "ConcatenationOptions",
        "AddOptions",
        "L2NormOptions",
        "LocalResponseNormalizationOptions",
        "LSTMOptions",
        "ResizeBilinearOptions",
        "CallOptions",
        "ReshapeOptions",
        "SkipGramOptions",
        "SpaceToDepthOptions",
        "EmbeddingLookupSparseOptions",
        "MulOptions",
        "PadOptions",
        "GatherOptions",
        "BatchToSpaceNDOptions",
        "SpaceToBatchNDOptions",
        "TransposeOptions",
        "ReducerOptions",
        "SubOptions",
        "DivOptions",
        "SqueezeOptions",
        "SequenceRNNOptions",
        "StridedSliceOptions",
        "ExpOptions",
        "TopKV2Options",
        "SplitOptions",
        "LogSoftmaxOptions",
        "CastOptions",
        "DequantizeOptions",
        "MaximumMinimumOptions",
        "ArgMaxOptions",
        "LessOptions",
        "NegOptions",
        "PadV2Options",
        "GreaterOptions",
        "GreaterEqualOptions",
        "LessEqualOptions",
        "SelectOptions",
        "SliceOptions",
        "TransposeConvOptions",
        "SparseToDenseOptions",
        "TileOptions",
        "ExpandDimsOptions",
        "EqualOptions",
        "NotEqualOptions",
        "ShapeOptions",
        "PowOptions",
        "ArgMinOptions",
        "FakeQuantOptions",
        "PackOptions",
        "LogicalOrOptions",
        "OneHotOptions",
        "LogicalAndOptions",
        "LogicalNotOptions",
        "UnpackOptions",
        "FloorDivOptions",
        "SquareOptions",
        "ZerosLikeOptions",
        "FillOptions",
        "BidirectionalSequenceLSTMOptions",
        "BidirectionalSequenceRNNOptions",
        "UnidirectionalSequenceLSTMOptions",
        "FloorModOptions",
        "RangeOptions",
        "ResizeNearestNeighborOptions",
        "LeakyReluOptions",
        "SquaredDifferenceOptions",
        "MirrorPadOptions",
        "AbsOptions",
        "SplitVOptions",
        "UniqueOptions",
        "ReverseV2Options",
        "AddNOptions",
        "GatherNdOptions",
        "CosOptions",
        "WhereOptions",
        "RankOptions",
        "ReverseSequenceOptions",
        "MatrixDiagOptions",
        "QuantizeOptions",
        "MatrixSetDiagOptions",
        "HardSwishOptions",
        "IfOptions",
        "WhileOptions",
        "DepthToSpaceOptions",
    ];

    for name in option_names {
        writeln!(&mut file, "{}\n", &BuiltinOptionsImpl { name },)?;
    }
    Ok(())
}

fn main() {
    import_stl_types();
    #[cfg(feature = "generate_model_apis")]
    {
        generate_memory_impl().unwrap();
        generate_vector_impl().unwrap();
        generate_builtin_options_impl().unwrap();
    }
    import_tflite_types();
    build_inline_cpp();
    if env::var("DOCS_RS").is_err() {
        prepare_tensorflow_library();
    }
}
