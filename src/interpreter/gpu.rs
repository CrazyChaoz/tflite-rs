//! GPU delegate for offloading parts of a model graph to the GPU.
//!
//! This module is only available with the `gpu` feature and requires a
//! TensorFlow Lite library that was built with GPU delegate support
//! (e.g. the `tensorflow-lite` package from nixpkgs).
//!
//! The `mod gpu;` declaration in `interpreter/mod.rs` is unconditional; the
//! `#![cfg]` below keeps this file empty when the `gpu` feature is off.
#![cfg(feature = "gpu")]

use std::ffi::{c_char, CString};
use std::ptr;

use crate::bindings;
use crate::interpreter::op_resolver::OpResolver;
use crate::interpreter::Interpreter;
use crate::{Error, Result};

cpp! {{
    #include "tensorflow/lite/delegates/gpu/delegate.h"
    #include "tensorflow/lite/delegates/gpu/delegate_options.h"
    #include "tensorflow/lite/interpreter.h"
}}

/// How the GPU delegate balances initialization latency against
/// throughput.
///
/// Mirrors `TfLiteGpuInferenceUsage`.
#[repr(i32)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferencePreference {
    /// The delegate will be used only once, therefore, bootstrap/init time
    /// should be taken into account.
    #[default]
    FastSingleAnswer = 0,
    /// Prefer maximizing the throughput. The same delegate will be used
    /// repeatedly on multiple inputs.
    SustainedSpeed = 1,
    /// Balance init latency and throughput. This results in slightly higher
    /// init latency than [`Self::FastSingleAnswer`] but inference latency
    /// closer to [`Self::SustainedSpeed`].
    Balanced = 2,
}

/// Ordered priorities that give better control over the desired semantics,
/// where `inference_priority1` is more important than
/// `inference_priority2` which in turn is more important than
/// `inference_priority3`.
///
/// Mirrors `TfLiteGpuInferencePriority`.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InferencePriority {
    /// Only valid when the more significant priorities are fully specified.
    Auto = 0,
    /// Never lower the precision (e.g. keep FP32 computation).
    MaxPrecision = 1,
    /// Prefer the lowest latency. May lower the precision (e.g. FP16
    /// computation).
    MinLatency = 2,
    /// Prefer the lowest memory usage.
    MinMemoryUsage = 3,
}

/// Bitmask of experimental flags for the GPU delegate.
///
/// Mirrors `TfLiteGpuExperimentalFlags`.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GpuExperimentalFlags(u64);

impl GpuExperimentalFlags {
    /// No experimental features enabled.
    pub const NONE: Self = Self(0);
    /// Enables inference on quantized models with the delegate.
    pub const ENABLE_QUANT: Self = Self(1 << 0);
    /// Enforces execution with the OpenCL backend.
    pub const CL_ONLY: Self = Self(1 << 1);
    /// Enforces execution with the OpenGL backend.
    pub const GL_ONLY: Self = Self(1 << 2);
    /// Enables serialization of GPU kernels & model data. Speeds up
    /// initialization at the cost of space on disk. Requires
    /// [`GpuDelegateOptions::serialization_dir`] and
    /// [`GpuDelegateOptions::model_token`] to be set.
    pub const ENABLE_SERIALIZATION: Self = Self(1 << 3);

    /// The raw bitmask value.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }
}

impl std::ops::BitOr for GpuExperimentalFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for GpuExperimentalFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl From<u64> for GpuExperimentalFlags {
    fn from(bits: u64) -> Self {
        Self(bits)
    }
}

/// Options for the GPU delegate.
///
/// Mirrors `TfLiteGpuDelegateOptionsV2`; the defaults match
/// `TfLiteGpuDelegateOptionsV2Default()`.
#[derive(Debug, Clone)]
pub struct GpuDelegateOptions {
    /// Allow the GPU to quantify tensors, downcast values or process in FP16
    /// to increase performance. For most models the precision loss is
    /// warranted.
    ///
    /// NOTE: obsolete upstream, precision is controlled via the inference
    /// priorities.
    pub is_precision_loss_allowed: bool,
    /// How the delegate balances init latency and throughput.
    pub inference_preference: InferencePreference,
    /// Most important priority.
    pub inference_priority1: InferencePriority,
    /// Second most important priority.
    pub inference_priority2: InferencePriority,
    /// Least important priority.
    pub inference_priority3: InferencePriority,
    /// Bitmask of experimental flags.
    pub experimental_flags: GpuExperimentalFlags,
    /// Limits the maximum number of graph partitions to be delegated.
    pub max_delegated_partitions: i32,
    /// Directory used for serialization, see
    /// [`GpuExperimentalFlags::ENABLE_SERIALIZATION`].
    pub serialization_dir: Option<String>,
    /// Unique token that namespaces all serialization entries, see
    /// [`GpuExperimentalFlags::ENABLE_SERIALIZATION`].
    pub model_token: Option<String>,
}

impl Default for GpuDelegateOptions {
    fn default() -> Self {
        Self {
            is_precision_loss_allowed: false,
            inference_preference: InferencePreference::FastSingleAnswer,
            inference_priority1: InferencePriority::MaxPrecision,
            inference_priority2: InferencePriority::Auto,
            inference_priority3: InferencePriority::Auto,
            experimental_flags: GpuExperimentalFlags::ENABLE_QUANT,
            max_delegated_partitions: 1,
            serialization_dir: None,
            model_token: None,
        }
    }
}

/// A delegate that offloads parts of a model graph to the GPU.
///
/// The delegate must outlive every [`Interpreter`] it is applied to, since
/// TFLite keeps a non-owning reference to it.
pub struct GpuDelegate {
    handle: *mut bindings::TfLiteDelegate,
}

impl GpuDelegate {
    /// Creates a new GPU delegate with the given options.
    ///
    /// Creating the delegate does not require a GPU: a compatible GPU
    /// backend is only probed when the delegate is applied to an
    /// interpreter, which fails with [`Error::InternalError`] if none is
    /// available.
    pub fn new(options: &GpuDelegateOptions) -> Result<Self> {
        let precision_loss = options.is_precision_loss_allowed as i32;
        let preference = options.inference_preference as i32;
        let priority1 = options.inference_priority1 as i32;
        let priority2 = options.inference_priority2 as i32;
        let priority3 = options.inference_priority3 as i32;
        let flags = options.experimental_flags.bits();
        let partitions = options.max_delegated_partitions;
        // Keep the `CString`s alive until after the delegate is created, so the
        // pointers handed to C are NUL-terminated and valid.
        let to_cstring = |s: &Option<String>, what: &str| -> Result<Option<CString>> {
            s.as_deref()
                .map(|s| {
                    CString::new(s).map_err(|_| {
                        Error::internal_error(format!("{what} contains an interior NUL byte"))
                    })
                })
                .transpose()
        };
        let serialization_dir_c = to_cstring(&options.serialization_dir, "serialization_dir")?;
        let model_token_c = to_cstring(&options.model_token, "model_token")?;
        let serialization_dir: *const c_char =
            serialization_dir_c.as_ref().map_or(ptr::null(), |s| s.as_ptr());
        let model_token: *const c_char =
            model_token_c.as_ref().map_or(ptr::null(), |s| s.as_ptr());

        #[allow(clippy::forgetting_copy_types, deprecated, clippy::transmute_num_to_bytes)]
        let handle = unsafe {
            cpp!([
                precision_loss as "int32_t",
                preference as "int32_t",
                priority1 as "int32_t",
                priority2 as "int32_t",
                priority3 as "int32_t",
                flags as "int64_t",
                partitions as "int32_t",
                serialization_dir as "const char*",
                model_token as "const char*"
            ] -> *mut bindings::TfLiteDelegate as "TfLiteDelegate*" {
                TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
                options.is_precision_loss_allowed = precision_loss;
                options.inference_preference = preference;
                options.inference_priority1 = priority1;
                options.inference_priority2 = priority2;
                options.inference_priority3 = priority3;
                options.experimental_flags = flags;
                options.max_delegated_partitions = partitions;
                options.serialization_dir = serialization_dir;
                options.model_token = model_token;
                return TfLiteGpuDelegateV2Create(&options);
            })
        };

        if handle.is_null() {
            return Err(Error::internal_error("failed to create GPU delegate"));
        }
        Ok(Self { handle })
    }
}

impl Drop for GpuDelegate {
    fn drop(&mut self) {
        let handle = self.handle;

        #[allow(clippy::forgetting_copy_types, deprecated)]
        unsafe {
            cpp!([handle as "TfLiteDelegate*"] {
                TfLiteGpuDelegateV2Delete(handle);
            });
        }
    }
}

impl<'a, Op> Interpreter<'a, Op>
where
    Op: OpResolver,
{
    /// Offloads the delegated parts of the model graph to the GPU.
    ///
    /// Must be called **before** [`Interpreter::allocate_tensors`]; build
    /// the interpreter with [`crate::InterpreterBuilder::build_unallocated`].
    ///
    /// `delegate` must outlive this interpreter: TFLite keeps a non-owning
    /// reference to it while the interpreter is alive.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InternalError`] if no compatible GPU backend
    /// (e.g. OpenCL) is available on the system or the graph cannot be
    /// delegated.
    pub fn modify_graph_with_delegate(&mut self, delegate: &GpuDelegate) -> Result<()> {
        let interpreter = self.handle_mut();
        let handle = delegate.handle;

        #[allow(clippy::forgetting_copy_types, deprecated)]
        let r = unsafe {
            cpp!([
                interpreter as "Interpreter*",
                handle as "TfLiteDelegate*"
            ] -> bool as "bool" {
                return interpreter->ModifyGraphWithDelegate(handle) == kTfLiteOk;
            })
        };
        if r {
            Ok(())
        } else {
            Err(Error::internal_error("failed to apply GPU delegate to the interpreter"))
        }
    }

    /// Offloads the delegated parts of the model graph to the GPU,
    /// **taking ownership** of `delegate`.
    ///
    /// The interpreter keeps the delegate alive for its entire lifetime, so
    /// there is no need to keep the [`GpuDelegate`] around.
    ///
    /// Must be called **before** [`Interpreter::allocate_tensors`]; build
    /// the interpreter with [`crate::InterpreterBuilder::build_unallocated`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InternalError`] if no compatible GPU backend
    /// (e.g. OpenCL) is available on the system or the graph cannot be
    /// delegated.
    pub fn modify_graph_with_delegate_owning(&mut self, delegate: GpuDelegate) -> Result<()> {
        let interpreter = self.handle_mut();
        let handle = delegate.handle;
        // The C++ interpreter takes ownership of the delegate from here on;
        // prevent `GpuDelegate::drop` from deleting it as well.
        std::mem::forget(delegate);

        #[allow(clippy::forgetting_copy_types, deprecated)]
        let r = unsafe {
            cpp!([
                interpreter as "Interpreter*",
                handle as "TfLiteDelegate*"
            ] -> bool as "bool" {
                return interpreter->ModifyGraphWithDelegate(
                    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
                        handle, TfLiteGpuDelegateV2Delete)) == kTfLiteOk;
            })
        };
        if r {
            Ok(())
        } else {
            Err(Error::internal_error("failed to apply GPU delegate to the interpreter"))
        }
    }
}
