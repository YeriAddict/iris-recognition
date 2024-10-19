import os
import cv2

DATASET_FOLDER = "data"

def load_dataset(folder_path):
    data = {"training": [], "testing": []}
    for eye_folder in os.listdir(folder_path):
        eye_path = os.path.join(folder_path, eye_folder)
        if os.path.isdir(eye_path):
            subfolders = sorted(os.listdir(eye_path))
            training_folder = os.path.join(eye_path, subfolders[0])
            testing_folder = os.path.join(eye_path, subfolders[1])
            
            for img_file in os.listdir(training_folder):
                img_path = os.path.join(training_folder, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    data["training"].append(image)

            for img_file in os.listdir(testing_folder):
                img_path = os.path.join(testing_folder, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    data["testing"].append(image)
    return data["training"], data["testing"]

def main():
    training, testing = load_dataset(DATASET_FOLDER)
    print("Training images:", len(training))
    print("Testing images:", len(testing))

if __name__ == "__main__":
    main()