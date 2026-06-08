# ============================================================
# TASK 04 - Hand Gesture Recognition Model
# Prodigy Infotech - Machine Learning Internship
# Dataset: https://www.kaggle.com/datasets/gti-upm/leapgestrecog
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import cv2
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow version: {tf.__version__}")

# ── 1. Configuration ──────────────────────────────────────────
IMG_SIZE   = 64
BATCH_SIZE = 32
EPOCHS     = 20

# ── 2. Auto Detect Dataset Path ───────────────────────────────
print("\nDetecting dataset path...")

images, labels = [], []

for root, dirs, files in os.walk('/kaggle/input'):
    img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if len(img_files) > 20:
        # Get label from parent folder name
        label = os.path.basename(root)
        for fname in img_files[:50]:  # 50 per folder
            img_path = os.path.join(root, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)

print(f"Total images loaded: {len(images)}")
print(f"Unique labels: {len(set(labels))}")
print(f"Labels: {sorted(set(labels))}")

X = np.array(images)
y_raw = np.array(labels)

# ── 3. Encode Labels ──────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(le.classes_)
print(f"\nNumber of classes: {num_classes}")

# ── 4. Visualize Sample Gestures ──────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, cls in enumerate(le.classes_[:10]):
    idx = np.where(y_raw == cls)[0][0]
    ax = axes[i // 5, i % 5]
    ax.imshow(X[idx], cmap='gray')
    ax.set_title(cls, fontsize=9)
    ax.axis('off')

plt.suptitle('Hand Gesture Classes', fontsize=14)
plt.tight_layout()
plt.savefig('task4_gestures.png', dpi=100)
plt.show()
print("Gesture samples saved!")

# ── 5. Preprocessing ──────────────────────────────────────────
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = to_categorical(y_encoded, num_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 6. Build CNN Model ────────────────────────────────────────
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ── 7. Train Model ────────────────────────────────────────────
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
]

print("\nTraining CNN model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ── 8. Training Curves ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('task4_training_curves.png', dpi=100)
plt.show()

# ── 9. Evaluate ───────────────────────────────────────────────
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, y_pred)
print(f"\n── Test Accuracy: {acc*100:.2f}% ──")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ── 10. Confusion Matrix ──────────────────────────────────────
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f'Confusion Matrix — Accuracy: {acc*100:.2f}%')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('task4_confusion_matrix.png', dpi=100)
plt.show()

# ── 11. Save Model ────────────────────────────────────────────
model.save('task4_gesture_model.h5')
print("\nModel saved as 'task4_gesture_model.h5'")
print("All outputs saved!")
