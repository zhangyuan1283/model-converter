from onnx_tf.backend import prepare
import onnx

# Load ONNX model
onnx_model = onnx.load("tracer2b.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("tracer2b_tf")
