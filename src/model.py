import tensorflow as tf
from tensorflow.keras import layers, models


def create_model():

    model = models.Sequential([

        # ===========================
        # Input Layer
        # ===========================
        layers.Input(shape=(48, 48, 1)),

        # ======================================================
        # Block 1
        # ======================================================
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ======================================================
        # Block 2
        # ======================================================
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ======================================================
        # Block 3
        # ======================================================
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # ======================================================
        # Classification Head
        # ======================================================
        layers.Flatten(),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),

        # ======================================================
        # Output Layer
        # ======================================================
        layers.Dense(7, activation='softmax')

    ])

    return model


if __name__ == "__main__":
    model = create_model()
    model.summary()