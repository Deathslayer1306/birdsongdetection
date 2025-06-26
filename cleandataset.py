import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load label classes
label_classes = np.load("/content/drive/MyDrive/dataset/cleaned_dataset/labels.npy", allow_pickle=True)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="/content/drive/MyDrive/bestfinalGMM.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Helper: remove silence
def remove_silence(y, sr, top_db=30):
    intervals = librosa.effects.split(y, top_db=top_db)
    y_nonsilent = np.concatenate([y[start:end] for start, end in intervals])
    return y_nonsilent

# Feature extraction using MFCC + GMM means
from sklearn.mixture import GaussianMixture
import noisereduce as nr

def extract_gmm_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=5.0)
        y = remove_silence(y, sr)
        y = nr.reduce_noise(y=y, sr=sr)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        gmm = GaussianMixture(n_components=16, covariance_type='diag', reg_covar=1e-3, random_state=42)
        gmm.fit(mfccs.T)
        gmm_features = gmm.means_.flatten()
        return gmm_features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prediction loop
test_dir = "/content/drive/MyDrive/dataset/cleaned_dataset/test"
y_true, y_pred = [], []

for class_name in os.listdir(test_dir):
    class_path = os.path.join(test_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        features = extract_gmm_features(file_path)
        if features is not None:
            input_data = np.array(features, dtype=np.float32).reshape(1, -1)
            input_data = np.expand_dims(input_data, axis=-1)  # Ensure the input shape matches the model's expected input

            # Check if the model's input shape matches
            input_shape = input_details[0]['shape']
            if input_data.shape != tuple(input_shape):
                print(f"Resizing input from {input_data.shape} to {input_shape}")
                input_data = np.resize(input_data, input_shape)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_index = np.argmax(output_data)
            y_pred.append(label_classes[predicted_index])
            y_true.append(class_name)

# Evaluation
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_classes))

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred, labels=label_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_classes, yticklabels=label_classes, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
