import os
import cv2
import numpy as np
from scipy.spatial import distance

from IrisLocalization import IrisLocalizer

# Global constants
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
LOCALIZED_FOLDER = "/localized"

class DataLoader:
    """
    Singleton class for loading training and testing data from a specified directory structure.

    Attributes
    ----------
        input_path (str): The path to the directory containing the data.
        training (list): A list of tuples containing training images and their file paths.
        testing (list): A list of tuples containing testing images and their file paths.

    Methods
    -------
        create() -> DataLoader:
            Creates and returns a singleton instance of the DataLoader class.
        load() -> tuple:
            Loads the training and testing data from the specified directory structure.
            Returns a tuple containing the training and testing data.
    """
    _instance = None

    def __init__(self):
        self.input_path = INPUT_FOLDER
        self.training = None
        self.testing = None

    @classmethod
    def create(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def load(self):
        data = {"training": [], "testing": []}
        for eye_folder in os.listdir(self.input_path):
            eye_path = os.path.join(self.input_path, eye_folder)
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

        self.training = data["training"]
        self.testing = data["testing"]

        return self.training, self.testing

class IrisRecognizer:
    def __init__(self, dataset):
        self.dataset = dataset

        self.input_path = INPUT_FOLDER
        self.output_path = OUTPUT_FOLDER
        self.localized_images_path = LOCALIZED_FOLDER

        # Iris Localization
        self.localized_images = []
        self.pupil_coordinates = []

    def localize_irises(self):
        for image, original_image_path in self.dataset:
            relative_path = os.path.relpath(original_image_path, self.input_path)
            localized_image_path = os.path.join(self.output_path + self.localized_images_path, relative_path)
            localized_image_directory = os.path.dirname(localized_image_path)
            os.makedirs(localized_image_directory, exist_ok=True)

            iris_localizer = IrisLocalizer(image)
            pupil_coordinates = iris_localizer.localize()
            iris_localizer.save_image(localized_image_path)

            self.localized_images.append((iris_localizer.image, original_image_path))
            self.pupil_coordinates.append(pupil_coordinates)
        return self.localized_images, self.pupil_coordinates
    
    def normalize_irises(self):
        pass

def main():
    training, testing = DataLoader.create().load()
    training_iris_recognizer = IrisRecognizer(training)
    localized_images, pupil_coordinates = training_iris_recognizer.localize_irises()

if __name__ == "__main__":
    main()