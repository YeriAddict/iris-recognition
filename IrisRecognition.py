import os
import cv2

from IrisLocalization import IrisLocalizer
from IrisNormalization import IrisNormalizer
from ImageEnhancement import IrisIlluminater, IrisEnhancer
from FeatureExtraction import FeatureExtractor
from IrisMatching import IrisMatcher
from PerformanceEvaluation import PerformanceEvaluator

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
        self.labels = []

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

    def extract_irises_features(self, mode):
        for enhanced_image, original_image_path in self.enhanced_images:
            feature_extractor = FeatureExtractor(enhanced_image)
            features = feature_extractor.extract_features()
            label = os.path.normpath(original_image_path).split(os.sep)[1]

            self.features_vectors += features
            if mode == "train":
                self.labels.append(label)
                self.labels.append(label)
                self.labels.append(label)
                self.labels.append(label)
                self.labels.append(label)
                self.labels.append(label)
                self.labels.append(label)
            elif mode == "test":
                self.labels.append(label)
            else:
                print("Warn: Wrong mode")


def main():
    training, testing = DataLoader.create().load()
    # Training
    training_iris_recognizer = IrisRecognizer(training)
    training_iris_recognizer.localize_irises()
    training_iris_recognizer.normalize_irises()
    training_iris_recognizer.illuminate_irises()
    training_iris_recognizer.enhance_irises()
    training_iris_recognizer.extract_irises_features("train")

    # Ouput of training_features_vectors: [[],[],...] (2268, )
    training_features_vectors = training_iris_recognizer.features_vectors
    training_labels = training_iris_recognizer.labels # (2268, )

    training_iris_matching = IrisMatcher(108)
    training_iris_matching.fit(training_features_vectors, training_labels)

    # Prediction
    # Testing
    testing_iris_recognizer = IrisRecognizer(testing)
    testing_iris_recognizer.localize_irises()
    testing_iris_recognizer.normalize_irises()
    testing_iris_recognizer.illuminate_irises()
    testing_iris_recognizer.enhance_irises()
    testing_iris_recognizer.extract_irises_features("test")

    # Ouput of testing_features_vectors: [[(vector1, angle1), (vector2, angle2), ...],[],[],...,[]]
    testing_features_vectors = testing_iris_recognizer.features_vectors
    testing_labels = testing_iris_recognizer.labels

    d1_predicting_labels = []
    d2_predicting_labels = []
    d3_predicting_labels = []

    # Prediction - matching
    for i in range(0, len(testing_features_vectors), 7):
        window = testing_features_vectors[i:i+7]
        best_d1, best_d2, best_d3 = float("inf"), float("inf"), float("inf")
        best_label_d1, best_label_d2, best_label_d3 = "", "", ""
        
        for v in window:
            d1_best_label, d1_best_dist = training_iris_matching.match(v, "L1")
            d2_best_label, d2_best_dist = training_iris_matching.match(v, "L2")
            d3_best_label, d3_best_dist = training_iris_matching.match(v, "COSINE")
            
            if d1_best_dist < best_d1:
                best_d1 = d1_best_dist
                best_label_d1 = d1_best_label
            if d2_best_dist < best_d2:
                best_d2 = d2_best_dist
                best_label_d2 = d2_best_label
            if d3_best_dist < best_d3:
                best_d3 = d3_best_dist
                best_label_d3 = d3_best_label

        d1_predicting_labels.append(best_label_d1)
        d2_predicting_labels.append(best_label_d2)
        d3_predicting_labels.append(best_label_d3)

    # Performance Check
    print("============= CRR:")
    n_classes = len(testing_labels)
    performance_evaluator = PerformanceEvaluator(n_classes)
    d1_crr = performance_evaluator.calculate_crr(testing_labels, d1_predicting_labels)
    d2_crr = performance_evaluator.calculate_crr(testing_labels, d2_predicting_labels)
    d3_crr = performance_evaluator.calculate_crr(testing_labels, d3_predicting_labels)
   
    print("L1 distance measure | ", d1_crr)
    print("L2 distance measure | ", d2_crr)
    print("Cosine distance measure | ", d3_crr)

    print(d1_predicting_labels[:10], "\n")
    print(testing_labels[:10])
    print(len(d1_predicting_labels), len(testing_labels)) # 432, 432

if __name__ == "__main__":
    main()