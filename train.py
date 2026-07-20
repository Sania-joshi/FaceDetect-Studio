import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model import create_model

# ===========================
# Configuration
# ===========================
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
EPOCHS = 30

# ===========================
# Load datasets
# ===========================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# ===========================
# Normalize pixel values
# ===========================
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

train_dataset = train_dataset.map(
    lambda x, y: (
        data_augmentation(normalization_layer(x), training=True),
        y
    )
)

val_dataset = val_dataset.map(
    lambda x, y: (normalization_layer(x), y)
)
#no changes b/c-This is important because validation data should represent real, untouched images.

# Speed up training
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
val_dataset = val_dataset.prefetch(AUTOTUNE)

# ===========================
# Build Model
# ===========================
model = create_model()

# ===========================
# Compile Model
# ===========================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005
)

model.compile(
    optimizer=optimizer,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ===========================
# Callbacks
# ===========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/best_emotion_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# ===========================
# Train Model
# ===========================
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint],
    verbose=2
)

# ===========================
# Save Final Model
# ===========================
model.save("models/emotion_model.keras")

print("\n✅ Model trained and saved successfully!")