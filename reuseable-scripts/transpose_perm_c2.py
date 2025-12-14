import onnx
import argparse

def modify_transpose_perm(model_path: str, output_path: str):
    # Load the ONNX model
    model = onnx.load(model_path)
    graph = model.graph


    input_to_consumers = {}
    for node in graph.node:
        for input_name in node.input:
            input_to_consumers.setdefault(input_name, []).append(node)

    modified_count = 0

    for node in graph.node:
        if node.op_type != "Transpose":
            continue

        transpose_output_names = list(node.output)
        feeds_matmul_second_input = False


        for out_name in transpose_output_names:
            consumers = input_to_consumers.get(out_name, [])
            for consumer in consumers:
                if consumer.op_type == "MatMul" and len(consumer.input) > 1 and consumer.input[1] == out_name:
                    feeds_matmul_second_input = True
                    break
            if feeds_matmul_second_input:
                break

        if not feeds_matmul_second_input:
            continue


        for attr in node.attribute:
            if attr.name == "perm":
                old_perm = list(attr.ints)
                if len(old_perm) >= 2:
                    new_perm = old_perm[:-2] + [old_perm[-1], old_perm[-2]]
                    attr.ints[:] = new_perm
                    modified_count += 1
                    print(f"Modified Transpose node '{node.name}': {old_perm} → {new_perm}")
                else:
                    print(f"Skipped '{node.name}': perm too short ({old_perm})")


    onnx.save(model, output_path)
    print(f"\n✅ Done! Modified {modified_count} Transpose node(s).")
    print(f"Saved modified model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Modify Transpose perm attributes in an ONNX model (swap last two dims if feeding MatMul input)."
    )
    parser.add_argument("input_model", help="Path to the input ONNX model")
    parser.add_argument("output_model", help="Path to save the modified ONNX model")
    args = parser.parse_args()

    modify_transpose_perm(args.input_model, args.output_model)

