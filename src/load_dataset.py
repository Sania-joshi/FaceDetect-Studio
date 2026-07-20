import tensorflow as tf #This imports the Deep Learning library.

# Image size
IMG_SIZE = (48, 48)

# Batch size
BATCH_SIZE = 32

# Training Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# Validation Dataset
val_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# Test Dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

print("\nDatasets loaded successfully!")

print("\nClass Names:")
print(train_dataset.class_names)