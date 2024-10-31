import os
import cv2
import random
import numpy as np

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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
    def __init__(self, training, testing, kernel_size, f, rotation_angles = [-9, -6, -3, 0, 3, 6, 9], n_classes=108, n_angles=7):
        # Public attributes
        self.training = training
        self.testing = testing
        self.kernel_size = kernel_size
        self.f = f 

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
    
    def verify(self, testing_pairs):
        # Calculate similarity scores for each pair (L1, L2, Cosine)
        similarity_scores = {
            "L1": [],
            "L2": [],
            "COSINE": []
        }

        for pair in testing_pairs:
            # Unpack the feature vectors from each pair
            feature_vector1, feature_vector2 = pair
            l1, l2, cos = self.__iris_matcher.match_pair(feature_vector1, feature_vector2)
            
            similarity_scores["L1"].append(l1)
            similarity_scores["L2"].append(l2)
            similarity_scores["COSINE"].append(cos)

        return similarity_scores


    def evaluate(self, testing_labels, predicted_labels):
        crr = {}
        crr["L1"] = self.__performance_evaluator.calculate_crr(testing_labels, predicted_labels["L1"])
        crr["L2"] = self.__performance_evaluator.calculate_crr(testing_labels, predicted_labels["L2"])
        crr["COSINE"] = self.__performance_evaluator.calculate_crr(testing_labels, predicted_labels["COSINE"])

        return crr  
    
def create_pairs(features, labels, num_genuine_pairs=1000, num_impostor_pairs=1000):
    """
    Creates genuine and impostor pairs from the given features and labels.
    
    Parameters:
        features (np.ndarray): Projected feature vectors (after applying LDA).
        labels (np.ndarray): Corresponding class labels for each feature vector.
        num_genuine_pairs (int): Number of genuine pairs to create.
        num_impostor_pairs (int): Number of impostor pairs to create.
    
    Returns:
        pairs (list of tuple): List of pairs of feature vectors.
        pair_labels (list of int): Labels for each pair (1 for genuine, 0 for impostor).
    """
    pairs = []
    pair_labels = []
    
    # Dictionary to group features by their label
    label_to_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
    
    # Generate Genuine Pairs
    for _ in range(num_genuine_pairs):
        label = random.choice(list(label_to_indices.keys()))
        i, j = np.random.choice(label_to_indices[label], 2, replace=False)
        pairs.append((features[i], features[j]))
        pair_labels.append(1)  # Label 1 for genuine pairs

    # Generate Impostor Pairs
    for _ in range(num_impostor_pairs):
        label1, label2 = np.random.choice(list(label_to_indices.keys()), 2, replace=False)
        i = np.random.choice(label_to_indices[label1])
        j = np.random.choice(label_to_indices[label2])
        pairs.append((features[i], features[j]))
        pair_labels.append(0)  # Label 0 for impostor pairs

    return pairs, pair_labels

def calculate_fmr_fnmr(similarity_scores, pair_labels, threshold):
    """
    Calculates False Match Rate (FMR) and False Non-Match Rate (FNMR) for a given threshold.
    
    Parameters:
        similarity_scores (list): List of similarity scores for the pairs.
        pair_labels (list): List of true labels for the pairs (1 for genuine, 0 for impostor).
        threshold (float): The threshold to evaluate the FMR and FNMR.
    
    Returns:
        fmr (float): False Match Rate.
        fnmr (float): False Non-Match Rate.
    """
    false_matches = 0
    false_non_matches = 0
    total_matches = 0
    total_non_matches = 0
    
    for score, label in zip(similarity_scores, pair_labels):
        if label == 0:  # Impostor pair
            total_non_matches += 1
            if score >= threshold:  # Incorrect match
                false_matches += 1
        else:  # Genuine pair
            total_matches += 1
            if score < threshold:  # Incorrect non-match
                false_non_matches += 1

    # Calculate FMR and FNMR
    fmr = false_matches / total_non_matches if total_non_matches > 0 else 0
    fnmr = false_non_matches / total_matches if total_matches > 0 else 0
    
    return fmr, fnmr

def main():
    # Tunable parameters
    kernel_size = 31
    f = 0.075
    thresholds = [0.446, 0.472, 0.502]         

    training, testing = DataLoader.create().load()

    iris_model = IrisRecognitionModel(training, testing, kernel_size, f)

    X_train, X_test, y_train, y_test = iris_model.extract_features_and_labels()

    iris_model.fit(X_train, y_train)

    # Identification mode
    # y_pred_identify = iris_model.identify(X_test)

    # crr = iris_model.evaluate(y_test, y_pred_identify)

    # print("L1 distance measure | ", crr["L1"])
    # print("L2 distance measure | ", crr["L2"])
    # print("Cosine distance measure | ", crr["COSINE"])

    # Verification mode
    # Generate pairs of features (genuine and impostor)
    num_genuine_pairs = 1000
    num_impostor_pairs = 1000
    pairs, pair_labels = create_pairs(X_test, np.array(y_test), num_genuine_pairs, num_impostor_pairs)

    y_pred_verify = iris_model.verify(pairs)

    # Calculate FMR and FNMR for each threshold per metric
    for threshold in thresholds:
        fmr, fnmr = calculate_fmr_fnmr(y_pred_verify["L1"], pair_labels, threshold)
        print(f"Metric: L1 | Threshold: {threshold} | FMR: {fmr:.4f} | FNMR: {fnmr:.4f}")

    for threshold in thresholds:
        fmr, fnmr = calculate_fmr_fnmr(y_pred_verify["L2"], pair_labels, threshold)
        print(f"Metric: L2 | Threshold: {threshold} | FMR: {fmr:.4f} | FNMR: {fnmr:.4f}")
    
    for threshold in thresholds:
        fmr, fnmr = calculate_fmr_fnmr(y_pred_verify["COSINE"], pair_labels, threshold)
        print(f"Metric: COSINE | Threshold: {threshold} | FMR: {fmr:.4f} | FNMR: {fnmr:.4f}")

    # Generate ROC curves for each similarity measure
    # for metric, scores in y_pred_verify.items():
    #     fpr, tpr, _ = roc_curve(pair_labels, scores, pos_label=1)
    #     roc_auc = auc(fpr, tpr)
    #     plt.plot(fpr, tpr, lw=2, label=f'{metric} (AUC = {roc_auc:.2f})')

    # # Plot ROC curve
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) - Verification Mode')
    # plt.legend(loc="lower right")
    # plt.show()

if __name__ == "__main__":
    main()