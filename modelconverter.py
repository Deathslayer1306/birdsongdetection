import tensorflow as tf

# Load a Keras model
model = tf.keras.models.load_model('/content/drive/MyDrive/bestfinal.keras')  # or .h5 file
# Convert model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the .tflite model to a file
tflite_model_path = "/content/drive/MyDrive/bestfinalGMM.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
