use maybe_owned::MaybeOwned;

use super::op_resolver::OpResolver;
use super::FlatBufferModel;
use super::Interpreter;
use crate::bindings::tflite as bindings;
use crate::{Error, Result};

cpp! {{
    #include "tensorflow/lite/model.h"
    #include "tensorflow/lite/kernels/register.h"

    using namespace tflite;
}}

pub struct InterpreterBuilder<'a, Op>
where
    Op: OpResolver,
{
    handle: Box<bindings::InterpreterBuilder>,
    _model: MaybeOwned<'a, FlatBufferModel>,
    _resolver: Op,
}

impl<'a, Op> Drop for InterpreterBuilder<'a, Op>
where
    Op: OpResolver,
{
    fn drop(&mut self) {
        let handle = Box::into_raw(std::mem::take(&mut self.handle));
        #[allow(clippy::forgetting_copy_types, clippy::useless_transmute, deprecated)]
        unsafe {
            cpp!([handle as "InterpreterBuilder*"] {
                delete handle;
            });
        }
    }
}

impl<'a, Op> InterpreterBuilder<'a, Op>
where
    Op: OpResolver,
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new<M: Into<MaybeOwned<'a, FlatBufferModel>>>(model: M, resolver: Op) -> Result<Self> {
        use std::ops::Deref;
        let model = model.into();
        let handle = {
            let model_handle = model.as_ref().handle.deref();
            let resolver_handle = resolver.get_resolver_handle();

            #[allow(clippy::forgetting_copy_types, deprecated)]
            unsafe {
                cpp!([model_handle as "const FlatBufferModel*",
                    resolver_handle as "const OpResolver*"
                ] -> *mut bindings::InterpreterBuilder as "InterpreterBuilder*" {
                    return new InterpreterBuilder(*model_handle, *resolver_handle);
                })
            }
        };
        if handle.is_null() {
            return Err(Error::InternalError("failed to create InterpreterBuilder".to_string()));
        }
        let handle = unsafe { Box::from_raw(handle) };
        Ok(Self { handle, _model: model, _resolver: resolver })
    }

    /// Builds the interpreter and allocates all tensors.
    ///
    /// This is equivalent to calling [`Self::build_unallocated`] followed
    /// by [`Interpreter::allocate_tensors`].
    pub fn build(mut self) -> Result<Interpreter<'a, Op>> {
        let handle = self.build_handle("failed to build")?;
        let mut interpreter = Interpreter::new(handle, self)?;
        interpreter.allocate_tensors()?;
        Ok(interpreter)
    }

    /// Builds the interpreter, using the given number of threads, and
    /// allocates all tensors.
    pub fn build_with_threads(
        mut self,
        threads: std::os::raw::c_int,
    ) -> Result<Interpreter<'a, Op>> {
        let handle = self.build_handle_with_threads(threads, "failed to build with threads")?;
        let mut interpreter = Interpreter::new(handle, self)?;
        interpreter.allocate_tensors()?;
        Ok(interpreter)
    }

    /// Builds the interpreter **without** allocating its tensors.
    ///
    /// Use this when a delegate (e.g. the GPU delegate from
    /// [`crate::gpu`]) has to be applied before the tensors are allocated.
    /// Call [`Interpreter::allocate_tensors`] before reading or writing
    /// tensors.
    pub fn build_unallocated(mut self) -> Result<Interpreter<'a, Op>> {
        let handle = self.build_handle("failed to build")?;
        Interpreter::new(handle, self)
    }

    fn build_handle(&mut self, error: &str) -> Result<*mut bindings::Interpreter> {
        #[allow(clippy::forgetting_copy_types, deprecated)]
        let handle = {
            let builder = &mut *self.handle;
            unsafe {
                cpp!([builder as "InterpreterBuilder*"] -> *mut bindings::Interpreter as "Interpreter*" {
                    std::unique_ptr<Interpreter> interpreter;
                    (*builder)(&interpreter);
                    return interpreter.release();
                })
            }
        };
        if handle.is_null() {
            return Err(Error::internal_error(error));
        }
        Ok(handle)
    }

    fn build_handle_with_threads(
        &mut self,
        threads: std::os::raw::c_int,
        error: &str,
    ) -> Result<*mut bindings::Interpreter> {
        #[allow(clippy::forgetting_copy_types, deprecated)]
        let handle = {
            let builder = &mut *self.handle;
            #[allow(clippy::transmute_num_to_bytes)]
            unsafe {
                cpp!([builder as "InterpreterBuilder*", threads as "int"] -> *mut bindings::Interpreter as "Interpreter*" {
                    std::unique_ptr<Interpreter> interpreter;
                    (*builder)(&interpreter, threads);
                    return interpreter.release();
                })
            }
        };
        if handle.is_null() {
            return Err(Error::internal_error(error));
        }
        Ok(handle)
    }
}
