#[cfg(feature = "generate_model_apis")]
#[macro_use]
extern crate bart_derive;

use std::env;
// use std::env::VarError; // legacy make feature no longer used
use std::path::{Path, PathBuf};
#[cfg(feature = "build")]
use std::time::Instant;
#[cfg(feature = "build")]
use std::process::Command;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
}

fn submodules() -> PathBuf {
    manifest_dir().join("submodules")
}

// Legacy make-based TensorFlow Lite build removed in favour of CMake.
// We now drive a CMake build directly from the TensorFlow submodule.
#[cfg(feature = "build")]
#[allow(clippy::too_many_lines)]
fn cmake_build_tensorflow() -> PathBuf {
    let start = Instant::now();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let build_dir = out_dir.join("tflite_cmake_build");
    let tf_lite_src = submodules().join("tensorflow/tensorflow/lite");
    std::fs::create_dir_all(&build_dir).expect("Unable to create CMake build dir");

    // Determine build type (Debug / Release)
    let build_type = if cfg!(feature = "debug_tflite") || cfg!(debug_assertions) {
        "Debug"
    } else {
        "Release"
    };

    // Only reconfigure if cache missing.
    // (Re)configure if no cache or no generated build files (e.g. prior failed attempt)
    let needs_configure = !build_dir.join("CMakeCache.txt").exists()
        || (!build_dir.join("Makefile").exists()
            && !build_dir.join("build.ninja").exists());
    if needs_configure {
        println!("Configuring TensorFlow Lite with CMake ({build_type})");
        let mut cfg = Command::new("cmake");
        cfg.current_dir(&build_dir);
        if let Ok(gen) = env::var("TFLITE_RS_CMAKE_GENERATOR") {
            if !gen.is_empty() { cfg.arg("-G").arg(gen); }
            println!("cargo:rerun-if-env-changed=TFLITE_RS_CMAKE_GENERATOR");
        }
        // Removed mut + unnecessary String allocations.
        let major = "2";
        let minor = "20";
        let patch = "0";
        //TF_MAJOR=2 TF_MINOR=20 TF_PATCH=0

        let defines = format!(
            "-DTF_MAJOR_VERSION={major} -DTF_MINOR_VERSION={minor} -DTF_PATCH_VERSION={patch} -DTF_VERSION_SUFFIX=''"
        );
        cfg.arg(format!("-DCMAKE_CXX_FLAGS={defines}"));
        cfg.arg(format!("-DCMAKE_C_FLAGS={defines}"));
        cfg.arg(&tf_lite_src)
            .arg(format!("-DCMAKE_BUILD_TYPE={build_type}"))
            .arg("-DCMAKE_POLICY_VERSION_MINIMUM=3.5");
        
        #[cfg(feature="gpu")]
        cfg.arg("-DTFLITE_ENABLE_GPU=ON");
        
        cfg.arg(format!("-Druy_DIR={build_dir:?}/ruy"));
        cfg.arg(format!("-Dxnnpack_DIR={build_dir:?}/xnnpack"));

        // Allow providing a toolchain file: TFLITE_RS_CMAKE_TOOLCHAIN_FILE
        if let Ok(toolchain) = env::var("TFLITE_RS_CMAKE_TOOLCHAIN_FILE") {
            if !toolchain.is_empty() {
                cfg.arg(format!("-DCMAKE_TOOLCHAIN_FILE={toolchain}"));
                println!("cargo:rerun-if-env-changed=TFLITE_RS_CMAKE_TOOLCHAIN_FILE");
            }
        }             


        // Pass through any -D variables via env prefixed TFLITE_RS_CMAKE_<NAME>
        for (k, v) in env::vars() {
            if let Some(raw) = k.strip_prefix("TFLITE_RS_CMAKE_") {
                if raw == "TOOLCHAIN_FILE" { continue; }
                // Skip empty values.
                if v.is_empty() { continue; }
                cfg.arg(format!("-D{raw}={v}"));
                println!("cargo:rerun-if-env-changed={k}");
            }
        }

        // Common toggles can be overridden via the env mechanism above.
        // Provide some gentle defaults if user did not set them.
        // (Don't force them if user supplied an override.)
        // Example defaults: enable XNNPACK (already ON by default), keep RUY OFF on non-Android.

        // Provide a default unless caller already provided override via env (we can't introspect args easily)
        // Users can override with TFLITE_RS_CMAKE_TFLITE_ENABLE_XNNPACK=OFF
        if env::var("TFLITE_RS_CMAKE_TFLITE_ENABLE_XNNPACK").is_err() {
            cfg.arg("-DTFLITE_ENABLE_XNNPACK=ON");
        }

    let status = cfg.status().expect("Failed to run cmake configuration for TensorFlow Lite");
    assert!(status.success(), "CMake configuration for TensorFlow Lite failed");
    }    
    
    // Build step.
    println!("Building TensorFlow Lite (cmake --build)");
    let mut build_cmd = Command::new("cmake");
    build_cmd.current_dir(&build_dir);
    build_cmd.arg("--build").arg(".");

    // Respect job server / explicit parallelism environment variables.
    // TFLITE_RS_CMAKE_PARALLELISM takes precedence, else fall back to NUM_JOBS provided by cargo.
    if let Ok(j) = env::var("TFLITE_RS_CMAKE_PARALLELISM") {
        if !j.is_empty() { build_cmd.arg("-j").arg(j); }
        println!("cargo:rerun-if-env-changed=TFLITE_RS_CMAKE_PARALLELISM");
    } else if let Ok(j) = env::var("NUM_JOBS") { // cargo sets this
        build_cmd.arg("-j").arg(j);
    }

    let status = build_cmd.status().expect("Failed to build TensorFlow Lite with CMake");
    assert!(status.success(), "CMake build for TensorFlow Lite failed");

    println!("CMake build completed in {:?}", start.elapsed());
    build_dir
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
    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    let arch_var = format!("TFLITE_{}_LIB_DIR", arch.replace('-', "_").to_uppercase());
    let all_var = "TFLITE_LIB_DIR".to_string();

    // If user supplies prebuilt location, use it. Else build via CMake.
    let supplied_lib_dir = env::var(&arch_var).ok().or_else(|| env::var(&all_var).ok());
    if let Some(lib_dir) = supplied_lib_dir {
        println!("cargo:rustc-link-search=native={lib_dir}");
        let static_dynamic = if Path::new(&lib_dir).join("libtensorflow-lite.a").exists() {
            "static"
        } else { "dylib" };
        println!("cargo:rustc-link-lib={static_dynamic}=tensorflow-lite");
        println!("cargo:rerun-if-changed={lib_dir}");
    } else {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let binary_changing_features = binary_changing_features();
        let desired_lib_name = format!("libtensorflow-lite{binary_changing_features}.a");
        let final_lib = out_dir.join(&desired_lib_name);
        if !final_lib.exists() {
            let build_dir = cmake_build_tensorflow();
            // Locate built primary lib
            let candidates = [
                build_dir.join("libtensorflow-lite.a"),
                build_dir.join("libtensorflow-lite.so"),
                build_dir.join("libtensorflow-lite.dylib"),
            ];
            let built = candidates.iter().find(|p| p.exists()).cloned()
                .unwrap_or_else(|| panic!("Unable to find built TensorFlow Lite library in {}", build_dir.display()));
            std::fs::copy(&built, &final_lib).unwrap_or_else(|e| panic!("Copy library failed: {e}"));

            // Also emit link directives for dependency static libs in build dir (best-effort)
            if let Ok(entries) = std::fs::read_dir(&build_dir) {
                for e in entries.flatten() {
                    let path = e.path();
                    if !path.is_file() { continue; }
                    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
                    if !name.starts_with("lib") { continue; }
                    if name == desired_lib_name { continue; }
                    if name.starts_with("libtensorflow-lite") { continue; }
                    if let Some(stripped) = name.strip_prefix("lib") {
                        if let Some(libname) = stripped.strip_suffix(".a") {
                            println!("cargo:rustc-link-lib=static={libname}");
                        } else if let Some(libname) = stripped.strip_suffix(".so") {
                            println!("cargo:rustc-link-lib=dylib={libname}");
                        } else if let Some(libname) = stripped.strip_suffix(".dylib") {
                            println!("cargo:rustc-link-lib=dylib={libname}");
                        }
                    }
                }
                println!("cargo:rustc-link-search=native={}", build_dir.display());
            }
        }
        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=tensorflow-lite{binary_changing_features}");
    }
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
}

// This generates "tflite_types.rs" containing structs and enums which are inter-operable with Glow.
fn import_tflite_types() {
    use bindgen::*;

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
        .replace("Vector_iterator","Vector_iterator<Data>")
        .replace("Vector_reverse_iterator","Vector_reverse_iterator<Data>")
        .replace("type _Rb_tree_insert_return_type","type _Rb_tree_insert_return_type<_Iterator,_NodeHandle>");
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
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++17")
        .clang_arg("-fms-extensions")
        .rustfmt_bindings(false)
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
