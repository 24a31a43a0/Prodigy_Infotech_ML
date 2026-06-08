# ============================================================
# TASK 03 - Cats vs Dogs Image Classification using SVM
# Prodigy Infotech - Machine Learning Internship
# Dataset: https://www.kaggle.com/datasets/tongpython/cat-and-dog
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
import warnings
warnings.filterwarnings('ignore')

# ── 1. Configuration ──────────────────────────────────────────
IMG_SIZE   = 64
MAX_IMAGES = 2000  # 1000 cats + 1000 dogs

# ── 2. Auto Detect Dataset Path ───────────────────────────────
print("Detecting dataset path...")
cat_dir = None
dog_dir = None

for root, dirs, files in os.walk('/kaggle/input'):
    for d in dirs:
        d_lower = d.lower()
        full_path = os.path.join(root, d)
        if 'cat' in d_lower and cat_dir is None:
            # Check if it contains images
            try:
                files_in = os.listdir(full_path)
                if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files_in[:5]):
                    cat_dir = full_path
            except:
                pass
        if 'dog' in d_lower and dog_dir is None:
            try:
                files_in = os.listdir(full_path)
                if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in files_in[:5]):
                    dog_dir = full_path
            except:
                pass

# If separate dirs not found, try single train directory
if cat_dir is None or dog_dir is None:
    for root, dirs, files in os.walk('/kaggle/input'):
        img_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(img_files) > 100:
            cat_files_check = [f for f in img_files if 'cat' in f.lower()]
            dog_files_check = [f for f in img_files if 'dog' in f.lower()]
            if cat_files_check and dog_files_check:
                cat_dir = root
                dog_dir = root
                print(f"Found mixed directory: {root}")
                break

print(f"Cat directory: {cat_dir}")
print(f"Dog directory: {dog_dir}")

# ── 3. Load Images ────────────────────────────────────────────
def load_images_from_dir(directory, label, max_count=1000, prefix=None):
    images, labels = [], []
    all_files = os.listdir(directory)
    img_files = [f for f in all_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if prefix:
        img_files = [f for f in img_files if f.lower().startswith(prefix)]
    
    img_files = img_files[:max_count]
    
    for fname in img_files:
        img_path = os.path.join(directory, fname)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img.flatten())
            labels.append(label)
    return images, labels

print("\nLoading images...")

if cat_dir == dog_dir:
    # Mixed directory
    cat_imgs, cat_labels = load_images_from_dir(cat_dir, 0, MAX_IMAGES//2, prefix='cat')
    dog_imgs, dog_labels = load_images_from_dir(dog_dir, 1, MAX_IMAGES//2, prefix='dog')
else:
    cat_imgs, cat_labels = load_images_from_dir(cat_dir, 0, MAX_IMAGES//2)
    dog_imgs, dog_labels = load_images_from_dir(dog_dir, 1, MAX_IMAGES//2)

X = np.array(cat_imgs + dog_imgs)
y = np.array(cat_labels + dog_labels)

print(f"Loaded: {X.shape[0]} images | Cats: {(y==0).sum()} | Dogs: {(y==1).sum()}")

# ── 4. Show Sample Images ─────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
cat_indices = np.where(y == 0)[0][:5]
dog_indices = np.where(y == 1)[0][:5]

for i, idx in enumerate(cat_indices):
    axes[0, i].imshow(X[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axes[0, i].set_title('Cat')
    axes[0, i].axis('off')

for i, idx in enumerate(dog_indices):
    axes[1, i].imshow(X[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    axes[1, i].set_title('Dog')
    axes[1, i].axis('off')

plt.suptitle('Sample Images from Dataset', fontsize=14)
plt.tight_layout()
plt.savefig('task3_samples.png', dpi=100)
plt.show()
print("Sample images saved!")

# ── 5. Preprocessing & PCA ────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nApplying PCA (100 components)...")
pca = PCA(n_components=100, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ── 6. Train/Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── 7. Train SVM ──────────────────────────────────────────────
print("\nTraining SVM (RBF kernel)... please wait...")
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
print("SVM trained successfully!")

# ── 8. Evaluate ───────────────────────────────────────────────
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n── Model Performance ──")
print(f"  Accuracy: {acc*100:.2f}%")
print(f"\n── Classification Report ──")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# ── 9. Confusion Matrix ───────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ['Predicted Cat', 'Predicted Dog'])
plt.yticks([0, 1], ['Actual Cat', 'Actual Dog'])
plt.title(f'Confusion Matrix\nAccuracy: {acc*100:.2f}%')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
plt.tight_layout()
plt.savefig('task3_confusion_matrix.png', dpi=100)
plt.show()
print("All outputs saved!")
