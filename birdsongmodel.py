import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import librosa
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import noisereduce as nr

# Silence removal function
def remove_silence(y, sr, top_db=30):
    if sr is None:
        raise ValueError("Sampling rate (sr) is None. Ensure audio is loaded correctly.")
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    y_non_silent = np.concatenate([y[start:end] for start, end in non_silent_intervals])
    return y_non_silent

# Feature extraction using GMM
def extract_features(file_path, n_components=16):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=5.0)
        y = remove_silence(y, sr)
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)  # Noise reduction

        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
        gmm.fit(mfccs)
        gmm_features = gmm.means_.flatten()

        return gmm_features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Load data function
def load_data(data_dir):
    X, y = [], []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(class_name)
    return np.array(X), np.array(y)

# Define focal loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, dtype='float32')
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss_value = -alpha_t * tf.pow((1 - p_t), gamma) * tf.math.log(p_t + tf.keras.backend.epsilon())
        return tf.reduce_mean(focal_loss_value)
    return focal_loss_fixed

# Dataset paths
dataset_path = '/content/drive/MyDrive/dataset/dataset/'
train_dir = dataset_path + 'train'
test_dir = dataset_path + 'test'
valid_dir = dataset_path + 'valid'

# Load datasets
X_train, y_train = load_data(train_dir)
X_valid, y_valid = load_data(valid_dir)
X_test, y_test = load_data(test_dir)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_valid_encoded = label_encoder.transform(y_valid)
y_test_encoded = label_encoder.transform(y_test)

# Reshape data for Conv1D input
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Oversample minority classes
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train_encoded)
X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], X_train.shape[1], 1)

# Build the model with Global Average Pooling and regularization
def build_improved_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(256, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        GlobalAveragePooling1D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.6),
        Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.6),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
    return model

# Initialize model
input_shape = (X_train.shape[1], 1)
num_classes = len(label_encoder.classes_)
model = build_improved_model(input_shape, num_classes)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_resampled),
    y=y_train_resampled
)
class_weights_dict = dict(enumerate(class_weights))

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_gmm.keras', monitor='val_loss', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history = model.fit(
    X_train_resampled, tf.keras.utils.to_categorical(y_train_resampled),
    epochs=100,
    batch_size=32,
    validation_data=(X_valid, tf.keras.utils.to_categorical(y_valid_encoded)),
    class_weight=class_weights_dict,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test_encoded))
print(f'Test Accuracy: {test_acc}')

# Plot confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test_encoded, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
