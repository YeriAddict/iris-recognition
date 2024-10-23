import os
import cv2
import numpy as np
from scipy.spatial import distance

from IrisLocalization import IrisLocalizer

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

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
                    data["training"].append((image, img_path))

            for img_file in os.listdir(testing_folder):
                img_path = os.path.join(testing_folder, img_file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    data["testing"].append((image, img_path))

    return data["training"], data["testing"]

def save_localized_images(dataset, folder_path, output_root_folder):
    for image, original_image_path in dataset:
        # Recreate the original folder structure under the new root folder 'localized'
        relative_path = os.path.relpath(original_image_path, folder_path)  # Get relative path to preserve structure
        localized_image_path = os.path.join(output_root_folder, relative_path)

        # Create the necessary directories in the output folder
        localized_image_dir = os.path.dirname(localized_image_path)
        os.makedirs(localized_image_dir, exist_ok=True)

        # Process and save the image
        iris_localizer = IrisLocalizer(image)
        pupil_coordinates = iris_localizer.localize_iris()
        iris_localizer.save_image(localized_image_path)

def main():
    training, testing = load_dataset(INPUT_FOLDER)
    save_localized_images(training, INPUT_FOLDER, OUTPUT_FOLDER + "/localized")

if __name__ == "__main__":
    main()