import onnx
import onnx.numpy_helper as np_helper
import numpy as np
import argparse

def modify_transpose_perm(input_model_path, output_model_path):
    """
    Loads an ONNX model, transposes the weights used in MatMul nodes (2D tensors),
    and saves the modified model.
    """

    model = onnx.load(input_model_path)


    for node in model.graph.node:
        if node.op_type == "MatMul":
            print(f"Processing MatMul node: {node.name}")


            for i, input_name in enumerate(node.input):
    
                if i != 1:
                    continue

             
                initializer = next((init for init in model.graph.initializer if init.name == input_name), None)
                if initializer is not None:
          
                    data = np_helper.to_array(initializer)
                    print(f"Original shape of '{initializer.name}': {data.shape}")

                    if data.ndim == 2:
                        transposed_data = data.T
                        initializer.CopyFrom(np_helper.from_array(transposed_data, initializer.name))
                        print(f"Transposed shape of '{initializer.name}': {transposed_data.shape}")
                    else:
                        print(f"Skipping '{initializer.name}' (not 2D, ndim={data.ndim})")

    onnx.save(model, output_model_path)
    print(f"Updated model saved as '{output_model_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transpose MatMul weight initializers in an ONNX model (only 2D weights)."
    )
    parser.add_argument("input_model", help="Path to the input ONNX model")
    parser.add_argument("output_model", help="Path to save the modified ONNX model")
    args = parser.parse_args()

    modify_transpose_perm(args.input_model, args.output_model)

