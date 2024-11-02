import os
import random
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

class IrisPipeline:
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

    def extract_irises_features(self, rotation_angles, kernel_size, f, mode):
        for enhanced_image, original_image_path in self.enhanced_images:
            feature_extractor = FeatureExtractor(enhanced_image, rotation_angles, kernel_size, f)
            features = feature_extractor.extract_features()
            label = os.path.normpath(original_image_path).split(os.sep)[1]

            self.features_vectors += features
            if mode == "Train":
                for _ in range(len(rotation_angles)):
                    self.labels.append(label)
            elif mode == "Test":
                self.labels.append(label)
            else:
                print("Warn: Wrong mode")

class IrisRecognitionModel:
    def __init__(self, training, testing, kernel_size, f, thresholds, rotation_angles = [-9, -6, -3, 0, 3, 6, 9], n_classes=108, n_angles=7):
        # Public attributes
        self.training = training
        self.testing = testing
        self.kernel_size = kernel_size
        self.f = f
        self.thresholds = thresholds

        # Private attributes
        self.__rotation_angles = rotation_angles
        self.__n_classes = n_classes
        self.__n_angles = n_angles
        self.__training_mode = "Train"
        self.__testing_mode = "Test"
        self.__iris_matcher = IrisMatcher(self.__n_classes)
        self.__performance_evaluator = PerformanceEvaluator(len(self.testing))

    def extract_features_and_labels(self):
        # Training
        training_iris_pipeline = IrisPipeline(self.training)
        training_iris_pipeline.localize_irises()
        training_iris_pipeline.normalize_irises()
        training_iris_pipeline.illuminate_irises()
        training_iris_pipeline.enhance_irises()
        training_iris_pipeline.extract_irises_features(self.__rotation_angles, self.kernel_size, self.f, self.__training_mode)
        training_features = training_iris_pipeline.features_vectors
        training_labels = training_iris_pipeline.labels

        # Testing
        testing_iris_pipeline = IrisPipeline(self.testing)
        testing_iris_pipeline.localize_irises()
        testing_iris_pipeline.normalize_irises()
        testing_iris_pipeline.illuminate_irises()
        testing_iris_pipeline.enhance_irises()
        testing_iris_pipeline.extract_irises_features(self.__rotation_angles, self.kernel_size, self.f, self.__testing_mode)
        testing_features = testing_iris_pipeline.features_vectors
        testing_labels = testing_iris_pipeline.labels

        return training_features, testing_features, training_labels, testing_labels

    def fit(self, training_features, training_labels):
        self.__iris_matcher.fit(training_features, training_labels)

    def identify(self, testing_features):
        predicted_labels = {}
        predicted_labels["L1"] = []
        predicted_labels["L2"] = []
        predicted_labels["COSINE"] = []

        for i in range(0, len(testing_features), self.__n_angles):
            class_features = testing_features[i:i+self.__n_angles]
            best_d1, best_d2, best_d3 = float("inf"), float("inf"), float("inf")
            best_d1_label, best_d2_label, best_d3_label = "", "", ""
            
            for features_vector in class_features:
                d1_label, d1 = self.__iris_matcher.match(features_vector, "L1")
                d2_label, d2 = self.__iris_matcher.match(features_vector, "L2")
                d3_label, d3 = self.__iris_matcher.match(features_vector, "COSINE")
                
                if d1 < best_d1:
                    best_d1 = d1
                    best_d1_label = d1_label
                if d2 < best_d2:
                    best_d2 = d2
                    best_d2_label = d2_label
                if d3 < best_d3:
                    best_d3 = d3
                    best_d3_label = d3_label

            predicted_labels["L1"].append(best_d1_label)
            predicted_labels["L2"].append(best_d2_label)
            predicted_labels["COSINE"].append(best_d3_label)

        return predicted_labels

    def verify(self, testing_features, testing_labels):
        verification_results = {}
        verification_results["COSINE"] = {}
        for threshold in self.thresholds:
            verification_results["COSINE"][threshold] = {"false_matches": 0, "false_non_matches": 0, "total_genuine_matches": 0, "total_impostor_matches": 0}

        new_testing_labels = []
        for k in range(len(testing_labels)):
            for _ in range(self.__n_angles):
                new_testing_labels.append(testing_labels[k])

        for features_vector, true_label in zip(testing_features, new_testing_labels):
            d3 = self.__iris_matcher.match(features_vector, "COSINE", true_label)

            for threshold in self.thresholds:

                if d3 > threshold:
                    verification_results["COSINE"][threshold]["false_non_matches"] += 1
                
                verification_results["COSINE"][threshold]["total_genuine_matches"] += 1

            impostor_labels = list(self.__iris_matcher.class_centers.keys())
            impostor_labels.remove(true_label)
            impostor_label = random.choice(impostor_labels)

            if impostor_label != true_label:
                d3_imposter = self.__iris_matcher.match(features_vector, "COSINE", claimed_class=impostor_label)

                for threshold in self.thresholds:

                    if d3_imposter < threshold:
                        verification_results["COSINE"][threshold]["false_matches"] += 1

                    verification_results["COSINE"][threshold]["total_impostor_matches"] += 1

        return verification_results

    def evaluate_crr(self, testing_labels, predicted_labels):
        crr = {}
        crr["L1"] = self.__performance_evaluator.calculate_crr(testing_labels, predicted_labels["L1"])
        crr["L2"] = self.__performance_evaluator.calculate_crr(testing_labels, predicted_labels["L2"])
        crr["COSINE"] = self.__performance_evaluator.calculate_crr(testing_labels, predicted_labels["COSINE"])

        return crr
    
    def evaluate_fmr_fnmr(self, verification_results):
        fmr = {}
        fmr["COSINE"] = {}
        
        fnmr = {}
        fnmr["COSINE"] = {}
        
        for threshold in self.thresholds:
            fmr["COSINE"][threshold] = self.__performance_evaluator.calculate_fmr(verification_results["COSINE"][threshold])
            fnmr["COSINE"][threshold] = self.__performance_evaluator.calculate_fnmr(verification_results["COSINE"][threshold])
        return fmr, fnmr

def main():
    # Tunable parameters
    kernel_size = 31 # Can be tuned
    f = 0.075 # Can be tuned
    thresholds = [0.155, 0.160, 0.165] # Can be tuned

    training, testing = DataLoader.create().load()

    iris_model = IrisRecognitionModel(training, testing, kernel_size, f, thresholds)

    X_train, X_test, y_train, y_test = iris_model.extract_features_and_labels()

    iris_model.fit(X_train, y_train)

    y_pred = iris_model.identify(X_test)

    y_verif = iris_model.verify(X_test, y_test)

    crr = iris_model.evaluate_crr(y_test, y_pred)

    fmr, fnmr = iris_model.evaluate_fmr_fnmr(y_verif)

    print("L1 distance measure     | ", round(crr["L1"], 4), "%")
    print("L2 distance measure     | ", round(crr["L2"], 4), "%")
    print("Cosine distance measure | ", round(crr["COSINE"], 4), "%")

    print("Cosine FMR  in format of (threshold: rate in %) | ", fmr["COSINE"])
    print("Cosine FNMR in format of (threshold: rate in %) | ", fnmr["COSINE"])

if __name__ == "__main__":
    main()