import tensorflow as tf
import matplotlib.pyplot as plt

IMG_SIZE = (48, 48)
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    shuffle=True
)

# Emotion labels
emotion_names = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# Take one batch
images, labels = next(iter(train_dataset))

print("Image Batch Shape :", images.shape)
print("Label Batch Shape :", labels.shape)

plt.figure(figsize=(10,10))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.imshow(images[i].numpy().astype("uint8").squeeze(), cmap="gray")

    plt.title(emotion_names[labels[i]])

    plt.axis("off")

plt.show()