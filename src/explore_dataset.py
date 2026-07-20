import os

dataset_path = "dataset"

for folder in ["train", "val", "test"]:

    print(f"\n{folder.upper()} DATASET")
    print("-" * 30)

    folder_path = os.path.join(dataset_path, folder)

    for emotion in sorted(os.listdir(folder_path)):

        emotion_path = os.path.join(folder_path, emotion)

        if os.path.isdir(emotion_path):
            total = len(os.listdir(emotion_path))
            print(f"Emotion {emotion} : {total} images")