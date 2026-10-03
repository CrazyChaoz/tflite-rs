#![cfg(feature = "gpu")]

use tflite::gpu::{GpuDelegate, GpuDelegateOptions};
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder, Result};

#[test]
fn create_gpu_delegate() -> Result<()> {
    // Creating the delegate does not require a GPU; it must succeed and be
    // cleanly dropped on any machine.
    let _delegate = GpuDelegate::new(&GpuDelegateOptions::default())?;
    Ok(())
}

#[test]
fn apply_gpu_delegate() -> Result<()> {
    let model = FlatBufferModel::build_from_file("data/MNISTnet_uint8_quant.tflite")?;
    let resolver = BuiltinOpResolver::default();

    let delegate = GpuDelegate::new(&GpuDelegateOptions::default())?;

    let builder = InterpreterBuilder::new(&model, &resolver)?;
    let mut interpreter = builder.build_unallocated()?;

    match interpreter.modify_graph_with_delegate(&delegate) {
        Ok(()) => {
            // A GPU backend is available: run inference on the delegated
            // graph.
            interpreter.allocate_tensors()?;

            assert_eq!(interpreter.inputs().len(), 1);
            let outputs = interpreter.outputs().to_vec();
            assert_eq!(outputs.len(), 1);

            interpreter.invoke()?;

            let output: &[u8] = interpreter.tensor_data(outputs[0])?;
            assert_eq!(output.len(), 10);
        }
        Err(error) => {
            // No compatible GPU backend on this machine; applying the
            // delegate must fail gracefully instead of panicking.
            eprintln!("no GPU backend available, skipping GPU inference: {error}");
        }
    }
    Ok(())
}
