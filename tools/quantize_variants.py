"""Smaller variants of fp32.onnx for size/accuracy comparison (not published)."""
import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

graph = onnx.load("fp32.onnx")
front_end = [n.name for n in graph.graph.node if n.op_type == "Conv" and "preprocessor" in n.name]
print("kept fp32:", front_end)
quantize_dynamic("fp32.onnx", "conv_int8.onnx", weight_type=QuantType.QUInt8,
                 per_channel=True, op_types_to_quantize=["MatMul", "Conv"],
                 nodes_to_exclude=front_end)
