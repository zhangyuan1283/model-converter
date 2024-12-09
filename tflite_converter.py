import tensorflow as tf

# Convert the TensorFlow model to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("tracer2b_tf")
tflite_model = converter.convert()

# Save the TFLite model
with open("tracer2b.tflite", "wb") as f:
    f.write(tflite_model)
