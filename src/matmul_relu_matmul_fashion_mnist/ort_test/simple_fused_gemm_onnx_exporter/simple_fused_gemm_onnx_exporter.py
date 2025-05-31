import onnx
from onnx import helper, TensorProto

# value infos
A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2,3])
B = helper.make_tensor_value_info("B", TensorProto.FLOAT, [3,4])
C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [2,4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2,4])

node = helper.make_node(
    "FusedGemm", ["A","B","C"], ["Y"],
    domain="com.microsoft", alpha=1.0, beta=1.0,
    transA=0, transB=0
)

graph = helper.make_graph([node], "fusedgemm_graph", [A,B,C], [Y])
# use com.microsoft opset version 1
model = helper.make_model(
    graph,
    opset_imports=[helper.make_opsetid("com.microsoft", 1)]
)
model.ir_version = 10
onnx.save(model, "fused_gemm.onnx")