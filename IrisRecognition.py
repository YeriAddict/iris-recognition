import os
import cv2
import numpy as np
from collections import Counter


from IrisLocalization import IrisLocalizer
from IrisNormalization import IrisNormalizer
from ImageEnhancement import IrisIlluminater, IrisEnhancer
from FeatureExtraction import FeatureExtractor
from IrisMatching import IrisMatcher

# Global constants
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
LOCALIZED_FOLDER = "/a_localized"
NORMALIZED_FOLDER = "/b_normalized"
ILLUMINATED_FOLDER = "/c_illuminated"
ENHANCED_FOLDER = "/d_enhanced"

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
        create() : DataLoader:
            Creates and returns a singleton instance of the DataLoader class.
        load() : tuple:
            Loads the training and testing data from the specified directory structure.
            Returns a tuple containing the training and testing data.
    """
    _instance = None

    def __init__(self):
        self.input_path = INPUT_FOLDER
        self.training = []
        self.testing = []

    @classmethod
    def create(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__init__()
        return cls._instance

    # def load(self):
    #     data = {"training": [], "testing": []}
    #     for eye_folder in os.listdir(self.input_path):
    #         eye_path = os.path.join(self.input_path, eye_folder)
    #         if os.path.isdir(eye_path):
    #             subfolders = sorted(os.listdir(eye_path))
    #             num_images = len(subfolders)
    #             # Ensure each class has enough images for training and testing
    #             if num_images < 2:
    #                 continue  # Skip classes with fewer than 2 images
    #             # Calculate the split for training and testing
    #             num_train = max(1, num_images // 3)  # Ensure at least one training image

    #             # Split into training and testing
    #             training_folder = subfolders[:num_train]
    #             testing_folder = subfolders[num_train:]
    #             # Load training images
    #             for img_file in training_folder:
    #                 img_path = os.path.join(eye_path, img_file)
    #                 image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #                 if image is not None:
    #                     data["training"].append((image, img_path))

    #             # Load testing images
    #             for img_file in testing_folder:
    #                 img_path = os.path.join(eye_path, img_file)
    #                 image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #                 if image is not None:
    #                     data["testing"].append((image, img_path))

    #     self.training = data["training"]
    #     self.testing = data["testing"]

    #     return self.training, self.testing
    def load(self):
        for eye_folder in os.listdir(self.input_path):
            eye_path = os.path.join(self.input_path, eye_folder)
            if os.path.isdir(eye_path):
                session_1_folder = os.path.join(eye_path, "1")
                session_2_folder = os.path.join(eye_path, "2")

                # Load exactly 3 images from session "1" for training
                if os.path.isdir(session_1_folder):
                    session_1_images = sorted(os.listdir(session_1_folder))
                    for img_file in session_1_images[:3]:  # Take only the first 3 images for training
                        img_path = os.path.join(session_1_folder, img_file)
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            self.training.append((image, img_path))

                # Load all images from session "2" for testing
                if os.path.isdir(session_2_folder):
                    for img_file in os.listdir(session_2_folder):
                        img_path = os.path.join(session_2_folder, img_file)
                        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            self.testing.append((image, img_path))
        print(f"Total training images: {len(self.training)}, Total testing images: {len(self.testing)}")  # Debugging line
        return self.training, self.testing

class IrisRecognizer:
    def __init__(self, dataset):
        self.dataset = dataset

        self.input_path = INPUT_FOLDER
        self.output_path = OUTPUT_FOLDER
        self.localized_images_path = LOCALIZED_FOLDER
        self.normalized_images_path = NORMALIZED_FOLDER
        self.illuminated_images_path = ILLUMINATED_FOLDER
        self.enhanced_images_path = ENHANCED_FOLDER

        # Iris Localization
        self.localized_images = []
        self.pupils_coordinates = []

        # Iris Normalization
        self.normalized_images = []

        # Iris Enhancement
        self.illuminated_images = []
        self.enhanced_images = []

        # Iris Feature Extraction
        self.features_vectors = []

    def localize_irises(self):
        for image, original_image_path in self.dataset:
            relative_path = os.path.relpath(original_image_path, self.input_path)
            localized_image_path = os.path.join(self.output_path + self.localized_images_path, relative_path)
            localized_image_directory = os.path.dirname(localized_image_path)
            os.makedirs(localized_image_directory, exist_ok=True)

            iris_localizer = IrisLocalizer(image)
            localized_image, pupil_coordinates = iris_localizer.localize_iris()
            iris_localizer.save_image(localized_image_path)

            self.localized_images.append((localized_image, original_image_path))
            self.pupils_coordinates.append(pupil_coordinates)
        return self.localized_images, self.pupils_coordinates
    
    def normalize_irises(self):
        for localized_image, pupil_coordinates in zip(self.localized_images, self.pupils_coordinates):
            image = localized_image[0]
            original_image_path = localized_image[1]

            relative_path = os.path.relpath(original_image_path, self.input_path)
            normalized_image_path = os.path.join(self.output_path + self.normalized_images_path, relative_path)
            normalized_image_directory = os.path.dirname(normalized_image_path)
            os.makedirs(normalized_image_directory, exist_ok=True)

            iris_normalizer = IrisNormalizer(image, pupil_coordinates)
            normalized_image = iris_normalizer.normalize_iris()
            iris_normalizer.save_image(normalized_image_path)

            self.normalized_images.append((normalized_image, original_image_path))
        return self.normalized_images
    
    def illuminate_irises(self):
        for normalized_image, original_image_path in self.normalized_images:
            relative_path = os.path.relpath(original_image_path, self.input_path)
            illuminated_image_path = os.path.join(self.output_path + self.illuminated_images_path, relative_path)
            illuminated_image_directory = os.path.dirname(illuminated_image_path)
            os.makedirs(illuminated_image_directory, exist_ok=True)

            iris_illuminater = IrisIlluminater(normalized_image)
            illuminated_image = iris_illuminater.illuminate_iris()
            iris_illuminater.save_image(illuminated_image_path)

            self.illuminated_images.append((illuminated_image, original_image_path))
        return self.illuminated_images        

    def enhance_irises(self):
        for normalized_image, illuminated_image in zip(self.normalized_images, self.illuminated_images):
            original_image_path = normalized_image[1]

            relative_path = os.path.relpath(original_image_path, self.input_path)
            enhanced_image_path = os.path.join(self.output_path + self.enhanced_images_path, relative_path)
            enhanced_image_directory = os.path.dirname(enhanced_image_path)
            os.makedirs(enhanced_image_directory, exist_ok=True)

            iris_enhancer = IrisEnhancer(normalized_image[0], illuminated_image[0])
            enhanced_image = iris_enhancer.enhance_iris()
            iris_enhancer.save_image(enhanced_image_path)

            self.enhanced_images.append((enhanced_image, original_image_path))
        return self.enhanced_images

    def extract_irises_features(self):
        for enhanced_image, original_image_path in self.enhanced_images:
            feature_extractor = FeatureExtractor(enhanced_image)
            features = feature_extractor.extract_features()

            self.features_vectors.append((features, original_image_path))
        return self.features_vectors

    def train_matcher(self):

        training_features = []
        training_labels = []

        for features, path in self.features_vectors:
            training_features.append(features)
            label = os.path.basename(os.path.dirname(path))
            training_labels.append(label)

        # Convert to numpy arrays
        training_features = np.array(training_features)
        training_labels = np.array(training_labels)

        # Train the matcher
        self.matcher = IrisMatcher()
        self.matcher.fit(training_features, training_labels)
    
    def match_irises(self, test_features):

        return self.matcher.match(test_features)

def main():
    training, testing = DataLoader.create().load()
    print(f"Loaded {len(training)} training images and {len(testing)} testing images.")

    training_iris_recognizer = IrisRecognizer(training)
    localized_images, pupil_coordinates = training_iris_recognizer.localize_irises()
    normalized_images = training_iris_recognizer.normalize_irises()
    illuminated_images = training_iris_recognizer.illuminate_irises()
    enhanced_images = training_iris_recognizer.enhance_irises()
    features_vectors = training_iris_recognizer.extract_irises_features()

    # Extract features for the testing images as well
    testing_iris_recognizer = IrisRecognizer(testing)
    testing_localized_images, testing_pupil_coordinates = testing_iris_recognizer.localize_irises()
    testing_normalized = testing_iris_recognizer.normalize_irises()
    testing_illuminated = testing_iris_recognizer.illuminate_irises()
    testing_enhanced = testing_iris_recognizer.enhance_irises()
    testing_features_vectors = testing_iris_recognizer.extract_irises_features()

    # Print out feature vectors length for debugging
    print(f"Extracted {len(features_vectors)} feature vectors.")

    # Prepare the features and labels for training
    training_features = [features for features, _ in features_vectors]
    training_labels = [os.path.basename(path).split('_')[0] for _, path in features_vectors]

    # Prepare the features and labels for testing
    testing_features = [features for features, _ in testing_features_vectors]
    testing_labels = [os.path.basename(path).split('_')[0] for _, path in testing_features_vectors]

    # # Print the class distribution in the training data
    class_counts = Counter(training_labels)
    # print("Class distribution in the training data:")
    # for class_label, count in class_counts.items():
    #     print(f"Class {class_label}: {count} samples")

    # Ensure at least two classes are present before proceeding with LDA
    # if len(class_counts) < 2:
    #     print("Error: The training data must contain at least two different classes for LDA.")
    #     return
    
    # Continue with the matching process if data is valid
    print(f"Total training images: {len(training_features)}, Total testing images: {len(testing_features)}")

    # Train the iris matcher and test the classification
    iris_matcher = IrisMatcher(num_classes=len(class_counts))
    iris_matcher.fit(training_features, training_labels)
    
    correct_matches = 0
    if len(testing_labels) > 0:
        for feature_vector, true_label in zip(testing_features, testing_labels):
            predicted_label, _ = iris_matcher.match(feature_vector)
            if predicted_label == true_label:
                correct_matches += 1

        accuracy = correct_matches / len(testing_labels)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    else:
        print("No testing labels available for evaluation.")

if __name__ == "__main__":
    main()