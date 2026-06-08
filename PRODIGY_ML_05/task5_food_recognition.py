# ============================================================
# TASK 05 - Food Recognition & Calorie Estimation
# Prodigy Infotech - Machine Learning Internship
# Dataset: https://www.kaggle.com/datasets/dansbecker/food-101
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow version: {tf.__version__}")
print("No internet required!")

# Configuration
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 10
MAX_PER_CLASS = 200

# Calorie Map
CALORIE_MAP = {
    'apple_pie': 237, 'baby_back_ribs': 320, 'baklava': 428,
    'beef_carpaccio': 150, 'beef_tartare': 196, 'beet_salad': 95,
    'beignets': 320, 'bibimbap': 210, 'bread_pudding': 280,
    'breakfast_burrito': 305,
}

# Correct path
images_dir = '/kaggle/input/datasets/dansbecker/food-101/food-101/food-101/images'

all_classes = sorted([d for d in os.listdir(images_dir)
    if os.path.isdir(os.path.join(images_dir, d))
    and not d.startswith('.')])[:NUM_CLASSES]

print(f"Total classes: {len(all_classes)}")
print(f"Using: {all_classes}")

for cls in all_classes:
    if cls not in CALORIE_MAP:
        CALORIE_MAP[cls] = 200

# Load Images
print("\nLoading images...")
images, labels = [], []

for cls in all_classes:
    cls_dir = os.path.join(images_dir, cls)
    img_files = [f for f in os.listdir(cls_dir)
        if f.endswith('.jpg') and not f.startswith('._')][:MAX_PER_CLASS]
    count = 0
    for fname in img_files:
        img_path = os.path.join(cls_dir, fname)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(cls)
                count += 1
        except:
            pass
    print(f"  {cls}: {count} images")

X = np.array(images, dtype='float32') / 255.0
y_raw = np.array(labels)
print(f"\nTotal images loaded: {len(X)}")

# Encode Labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
y = to_categorical(y_encoded, NUM_CLASSES)
class_names = list(le.classes_)

# Sample Food Images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, cls in enumerate(all_classes[:10]):
    idx = np.where(y_raw == cls)[0]
    if len(idx) > 0:
        ax = axes[i // 5, i % 5]
        ax.imshow(X[idx[0]])
        cal = CALORIE_MAP.get(cls, 200)
        ax.set_title(f'{cls}\n~{cal} kcal', fontsize=8)
        ax.axis('off')
plt.suptitle('Sample Food Images with Calorie Info', fontsize=13)
plt.tight_layout()
plt.savefig('task5_food_samples.png', dpi=100)
plt.show()
print("Food samples saved!")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(0.001),
    loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
]

print("\nTraining...")
history = model.fit(X_train, y_train, validation_split=0.15,
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

# Training Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'], label='Train')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[1].plot(history.history['loss'], label='Train')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].legend()
plt.tight_layout()
plt.savefig('task5_training_curves.png', dpi=100)
plt.show()

# Evaluate
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)
acc = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(classification_report(y_true, y_pred, target_names=class_names))

# Calorie Chart
plt.figure(figsize=(12, 5))
cals = [CALORIE_MAP.get(c, 200) for c in class_names]
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(class_names)))
bars = plt.bar(class_names, cals, color=colors, edgecolor='white')
plt.xticks(rotation=30, ha='right')
plt.ylabel('Calories per 100g')
plt.title('Calorie Content of Recognized Food Items')
for bar, cal in zip(bars, cals):
    plt.text(bar.get_x() + bar.get_width()/2,
        bar.get_height() + 3, str(cal), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('task5_calorie_chart.png', dpi=100)
plt.show()

# Save
model.save('task5_food_recognition_model.h5')
with open('task5_calorie_map.json', 'w') as f:
    json.dump(CALORIE_MAP, f, indent=2)
print("\nAll outputs saved successfully!")
