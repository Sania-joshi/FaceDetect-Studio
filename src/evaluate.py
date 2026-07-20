import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ===========================
# Configuration
# ===========================

IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# ===========================
# Load Validation Dataset
# ===========================

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=False
)

# Save class names
class_names = val_dataset.class_names

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

val_dataset = val_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)

# ===========================
# Load Model
# ===========================

model = tf.keras.models.load_model(
    "models/best_emotion_model.keras"
)

print("✅ Model Loaded Successfully!")

# ===========================
# Predict
# ===========================

predictions = model.predict(val_dataset)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.concatenate(
    [y.numpy() for x, y in val_dataset],
    axis=0
)

# ===========================
# Accuracy
# ===========================

accuracy = np.mean(predicted_labels == true_labels)

print(f"\nAccuracy: {accuracy*100:.2f}%")

# ===========================
# Classification Report
# ===========================

print("\nClassification Report\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )
)

# ===========================
# Confusion Matrix
# ===========================

cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(8,6))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=45
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

plt.show()