import os
import random
import cv2
from matplotlib import pyplot as plt
import numpy as np

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
        self.training = None
        self.testing = None

        self.__input_path = INPUT_FOLDER

    @classmethod
    def create(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def load(self):
        """
        Loads training and testing images from subfolders within the input path.
        The images are read in grayscale mode and stored in a dictionary with keys "training" and "testing".
        
        Returns:
            tuple: A tuple containing two lists:
                - training (list): A list of tuples where each tuple contains a training image and its file path.
                - testing (list): A list of tuples where each tuple contains a testing image and its file path.
        """
        data = {"training": [], "testing": []}
        for eye_folder in os.listdir(self.__input_path):
            eye_path = os.path.join(self.__input_path, eye_folder)
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
    """
    Class for processing iris images in a dataset: localization, normalization, illumination correction, enhancement, and feature extraction.
    It uses the IrisLocalizer, IrisNormalizer, IrisIlluminater, IrisEnhancer, and FeatureExtractor classes.

    Attributes:
        dataset : list
            A list of tuples containing images and their original paths.
        input_path : str
            Path to the input folder containing the original images.
        output_path : str
            Path to the output folder where processed images will be saved.
        localized_images_path : str
            Path to the folder where localized images will be saved.
        normalized_images_path : str
            Path to the folder where normalized images will be saved.
        illuminated_images_path : str
            Path to the folder where illuminated images will be saved.
        enhanced_images_path : str
            Path to the folder where enhanced images will be saved.
        localized_images : list
            A list to store localized images and their original paths.
        pupils_coordinates : list
            A list to store coordinates of pupils in the localized images.
        normalized_images : list
            A list to store normalized images and their original paths.
        illuminated_images : list
            A list to store illuminated images and their original paths.
        enhanced_images : list
            A list to store enhanced images and their original paths.
        features_vectors : list
            A list to store feature vectors extracted from the enhanced images.
        labels : list
            A list to store labels corresponding to the feature vectors.
    Methods:
        localize_irises():
            Localizes the iris in each image in the dataset and saves the localized images.
        normalize_irises():
            Normalizes the localized iris images and saves the normalized images.
        illuminate_irises():
            Applies illumination correction to the normalized iris images and saves the illuminated images.
        enhance_irises():
            Enhances the illuminated iris images and saves the enhanced images.
        extract_irises_features(rotation_angles, kernel_size, f, mode):
            Extracts features from the enhanced iris images and stores them along with their labels.
    """

    def __init__(self, dataset):
        self.dataset = dataset

        # Iris Feature Extraction
        self.features_vectors = []
        self.labels = []

        self.__input_path = INPUT_FOLDER
        self.__output_path = OUTPUT_FOLDER
        self.__localized_images_path = LOCALIZED_FOLDER
        self.__normalized_images_path = NORMALIZED_FOLDER
        self.__illuminated_images_path = ILLUMINATED_FOLDER
        self.__enhanced_images_path = ENHANCED_FOLDER

        # Iris Localization
        self.__localized_images = []
        self.__pupils_coordinates = []

        # Iris Normalization
        self.__normalized_images = []

        # Iris Enhancement
        self.__illuminated_images = []
        self.__enhanced_images = []

    def localize_irises(self):
        """
        Localizes irises in the images from the dataset and saves the localized images using the IrisLocalizer class.
        It saves the localized images in the folder self.__output_folder + "/" + self.__localized_images_path
        """
        for image, original_image_path in self.dataset:
            # Creates the path to where the localized image will be saved
            relative_path = os.path.relpath(original_image_path, self.__input_path)
            localized_image_path = os.path.join(self.__output_path + self.__localized_images_path, relative_path)
            localized_image_directory = os.path.dirname(localized_image_path)
            os.makedirs(localized_image_directory, exist_ok=True)

            # Localize the iris in the image
            iris_localizer = IrisLocalizer(image)
            localized_image, pupil_coordinates = iris_localizer.localize_iris()
            iris_localizer.save_image(localized_image_path)

            # Store the localized image and pupil coordinates
            self.__localized_images.append((localized_image, original_image_path))
            self.__pupils_coordinates.append(pupil_coordinates)
    
    def normalize_irises(self):
        """
        Normalizes irises from the localized images list and saves the normalized images using the IrisNormalizer class.
        It saves the normalized images in the folder self.__output_folder + "/" + self.__normalized_images_path
        """        
        for localized_image, pupil_coordinates in zip(self.__localized_images, self.__pupils_coordinates):
            image = localized_image[0]
            original_image_path = localized_image[1]

            # Creates the path to where the normalized image will be saved
            relative_path = os.path.relpath(original_image_path, self.__input_path)
            normalized_image_path = os.path.join(self.__output_path + self.__normalized_images_path, relative_path)
            normalized_image_directory = os.path.dirname(normalized_image_path)
            os.makedirs(normalized_image_directory, exist_ok=True)

            # Normalize the iris in the localized image
            iris_normalizer = IrisNormalizer(image, pupil_coordinates)
            normalized_image = iris_normalizer.normalize_iris()
            iris_normalizer.save_image(normalized_image_path)

            # Store the normalized image
            self.__normalized_images.append((normalized_image, original_image_path))
    
    def illuminate_irises(self):
        """
        Creates background illuminated irises from the normalized images list and saves them using the IrisIlluminater class.
        It saves the background illuminated images in the folder self.__output_folder + "/" + self.__illuminated_images_path.
        """           
        for normalized_image, original_image_path in self.__normalized_images:
            # Creates the path to where the illuminated image will be saved
            relative_path = os.path.relpath(original_image_path, self.__input_path)
            illuminated_image_path = os.path.join(self.__output_path + self.__illuminated_images_path, relative_path)
            illuminated_image_directory = os.path.dirname(illuminated_image_path)
            os.makedirs(illuminated_image_directory, exist_ok=True)

            # Illuminate the iris in the normalized image
            iris_illuminater = IrisIlluminater(normalized_image)
            illuminated_image = iris_illuminater.illuminate_iris()
            iris_illuminater.save_image(illuminated_image_path)

            # Store the illuminated image
            self.__illuminated_images.append((illuminated_image, original_image_path))    

    def enhance_irises(self):
        """
        Enhances irises from the normalized and background illuminated images list and saves them using the IrisEnhancer class.
        It saves the enhanced images in the folder self.__output_folder + "/" + self.__enhanced_images_path.
        """
        for normalized_image, illuminated_image in zip(self.__normalized_images, self.__illuminated_images):
            original_image_path = normalized_image[1]

            # Creates the path to where the enhanced image will be saved
            relative_path = os.path.relpath(original_image_path, self.__input_path)
            enhanced_image_path = os.path.join(self.__output_path + self.__enhanced_images_path, relative_path)
            enhanced_image_directory = os.path.dirname(enhanced_image_path)
            os.makedirs(enhanced_image_directory, exist_ok=True)

            # Enhance the iris
            iris_enhancer = IrisEnhancer(normalized_image[0], illuminated_image[0])
            enhanced_image = iris_enhancer.enhance_iris()
            iris_enhancer.save_image(enhanced_image_path)

            # Store the enhanced image
            self.__enhanced_images.append((enhanced_image, original_image_path))

    def extract_irises_features(self, rotation_angles, kernel_size, f, mode):
        """
        Extracts features from enhanced iris images and assigns labels based on the mode.
        
        Parameters:
            rotation_angles (list): A list of rotation angles to be used to rotate images.
            kernel_size (int): The size of the custom Gabor kernel.
            f (float): The frequency parameter for the custom Gabor kernel.
            mode (str): In "Train" mode, labels are appended for each rotated image. In "Test" mode, a single label is appended for each image.
        
        Raises:
            ValueError: If an invalid mode is provided.        
        """
        for enhanced_image, original_image_path in self.__enhanced_images:
            # Extract features from the enhanced image and its label
            feature_extractor = FeatureExtractor(enhanced_image, rotation_angles, kernel_size, f)
            features = feature_extractor.extract_features()
            label = os.path.normpath(original_image_path).split(os.sep)[1]

            # Store the features and labels
            self.features_vectors += features
            if mode == "Train":
                for _ in range(len(rotation_angles)):
                    self.labels.append(label)
            elif mode == "Test":
                self.labels.append(label)
            else:
                raise ValueError("Invalid mode")

class IrisRecognitionModel:
    """
    Class for the built Iris Recognition Model.
    It uses the IrisPipeline, IrisMatcher, and PerformanceEvaluator classes.

    Attributes
    ----------
        training : list
            A list containing the training data.
        testing : list
            A list containing the testing data.
        kernel_size : int
            The size of the custom Gabor kernel.
        f : float
            The frequency parameter for the custom Gabor kernel.
        thresholds : list
            A list of thresholds used for verification mode to calculate fmr and fnmr.
        rotation_angles : list, optional
            A list of angles for rotating the iris images (default is [-9, -6, -3, 0, 3, 6, 9]).
        n_classes : int, optional
            The number of classes in the dataset (default is 108).
        n_angles : int, optional
            The number of angles used in feature extraction (default is 7).
    
    Methods
    -------
        extract_features_and_labels():
            Extracts features and labels from the training and testing data.
        fit(training_features, training_labels):
            Fits the model using the training features and labels.
        identify(testing_features):
            Identification mode: Identifies the class labels for the given testing features.
        verify(testing_features, testing_labels):
            Verification mode: Verifies the testing features against the true labels.
        evaluate_crr(testing_labels, predicted_labels):
            Evaluates the Correct Recognition Rate (CRR) using the testing and predicted labels.
        evaluate_fmr_fnmr(verification_results):
            Evaluates the False Match Rate (FMR) and False Non-Match Rate (FNMR) using the verification results.
    """

    def __init__(self, training, testing, kernel_size, f, thresholds, rotation_angles = [-9, -6, -3, 0, 3, 6, 9], n_classes=108, n_angles=7):
        self.__training = training
        self.__testing = testing
        self.__kernel_size = kernel_size
        self.__f = f
        self.__thresholds = thresholds
        self.__rotation_angles = rotation_angles
        self.__n_classes = n_classes
        self.__n_angles = n_angles
        self.__training_mode = "Train"
        self.__testing_mode = "Test"
        self.__iris_matcher = IrisMatcher(self.__n_classes)
        self.__performance_evaluator = PerformanceEvaluator(len(self.__testing))

    def extract_features_and_labels(self):
        """
        Extracts features and labels for both training and testing datasets.

        Returns:
            tuple: A tuple containing:
                - training_features (list): The feature vectors for the training dataset.
                - testing_features (list): The feature vectors for the testing dataset.
                - training_labels (list): The labels for the training dataset.
                - testing_labels (list): The labels for the testing dataset.
        """
        # Training
        training_iris_pipeline = IrisPipeline(self.__training)
        training_iris_pipeline.localize_irises()
        training_iris_pipeline.normalize_irises()
        training_iris_pipeline.illuminate_irises()
        training_iris_pipeline.enhance_irises()
        training_iris_pipeline.extract_irises_features(self.__rotation_angles, self.__kernel_size, self.__f, self.__training_mode)
        training_features = training_iris_pipeline.features_vectors
        training_labels = training_iris_pipeline.labels

        # Testing
        testing_iris_pipeline = IrisPipeline(self.__testing)
        testing_iris_pipeline.localize_irises()
        testing_iris_pipeline.normalize_irises()
        testing_iris_pipeline.illuminate_irises()
        testing_iris_pipeline.enhance_irises()
        testing_iris_pipeline.extract_irises_features(self.__rotation_angles, self.__kernel_size, self.__f, self.__testing_mode)
        testing_features = testing_iris_pipeline.features_vectors
        testing_labels = testing_iris_pipeline.labels

        return training_features, testing_features, training_labels, testing_labels

    def fit(self, training_features, training_labels):
        """
        Trains the iris recognition model using the provided training features and labels.

        Parameters:
            training_features (list): The features of the training data.
            training_labels (list): The labels corresponding to the training data.
        """
        self.__iris_matcher.fit(training_features, training_labels)

    def identify(self, testing_features):
        """
        Identifies the class labels for the given testing features using different distance metrics (L1, L2 and Cosine).
        
        Parameters:
            testing_features (list): A list of feature vectors to be classified.
        
        Returns:
            dict: A dictionary containing the predicted labels for each distance metric.
                The keys are "L1", "L2", and "COSINE", and the values are lists of predicted labels.
        """
        predicted_labels = {}
        predicted_labels["L1"] = []
        predicted_labels["L2"] = []
        predicted_labels["COSINE"] = []

        for i in range(0, len(testing_features), self.__n_angles):
            # Extract a sublist of n_angles number of features vectors for each class
            class_features = testing_features[i:i+self.__n_angles]
            best_d1, best_d2, best_d3 = float("inf"), float("inf"), float("inf")
            best_d1_label, best_d2_label, best_d3_label = "", "", ""
            
            # Match each feature vector to the class centers
            for features_vector in class_features:
                d1_label, d1 = self.__iris_matcher.match(features_vector, "L1")
                d2_label, d2 = self.__iris_matcher.match(features_vector, "L2")
                d3_label, d3 = self.__iris_matcher.match(features_vector, "COSINE")
                
                # Update the best distance and label for each distance metric
                if d1 < best_d1:
                    best_d1 = d1
                    best_d1_label = d1_label
                if d2 < best_d2:
                    best_d2 = d2
                    best_d2_label = d2_label
                if d3 < best_d3:
                    best_d3 = d3
                    best_d3_label = d3_label

            # Append the best label for each distance metric
            predicted_labels["L1"].append(best_d1_label)
            predicted_labels["L2"].append(best_d2_label)
            predicted_labels["COSINE"].append(best_d3_label)

        return predicted_labels

    def verify(self, testing_features, testing_labels):
        """
        Verify the iris recognition results using cosine similarity.
        
        Parameters:
            testing_features (list): A list of feature vectors for the testing dataset.
            testing_labels (list): A list of true labels corresponding to the testing dataset.
        
        Returns:
            dict: A dictionary containing the verification results for cosine distance. 
                The key is "COSINE", and the values are dictionaries containing the verification results for each threshold.
        """
        verification_results = {}
        verification_results["COSINE"] = {}
        for threshold in self.__thresholds:
            verification_results["COSINE"][threshold] = {"false_matches": 0, "false_non_matches": 0, "total_genuine_matches": 0, "total_impostor_matches": 0}

        # Duplicate the testing labels for each angle so that the number of labels matches the number of features vectors
        new_testing_labels = []
        for k in range(len(testing_labels)):
            for _ in range(self.__n_angles):
                new_testing_labels.append(testing_labels[k])

        for features_vector, true_label in zip(testing_features, new_testing_labels):
            # Calculate the distance to the true class center
            d3 = self.__iris_matcher.match(features_vector, "COSINE", true_label)

            for threshold in self.__thresholds:
                # Increment the count when a genuine match is counted as non-match
                if d3 > threshold:
                    verification_results["COSINE"][threshold]["false_non_matches"] += 1
                
                # Increment to the count of total number of genuine attempts
                verification_results["COSINE"][threshold]["total_genuine_matches"] += 1

            # Choose an imposter label that is different from the true label
            impostor_labels = list(self.__iris_matcher.class_centers.keys())
            impostor_labels.remove(true_label)
            impostor_label = random.choice(impostor_labels)
            if impostor_label != true_label:
                # Calculate the distance to the true class center
                d3_imposter = self.__iris_matcher.match(features_vector, "COSINE", claimed_class=impostor_label)

                for threshold in self.__thresholds:
                    # Increment the count when an imposter match is counted as a match
                    if d3_imposter < threshold:
                        verification_results["COSINE"][threshold]["false_matches"] += 1

                    # Increment to the count of total number of imposter attempts
                    verification_results["COSINE"][threshold]["total_impostor_matches"] += 1

        return verification_results

    def evaluate_crr(self, testing_labels, predicted_labels):
        """
        Evaluates the Correct Recognition Rate (CRR) for different distance metrics.

        Parameters:
            testing_labels (list): The true labels of the testing dataset.
            predicted_labels (dict): A dictionary containing the predicted labels for each distance metric.
                The keys are "L1", "L2", and "COSINE", and the values are lists of predicted labels.

        Returns:
            dict: A dictionary with the CRR values for each distance metric ("L1", "L2", "COSINE").
        """
        crr = {}
        crr["L1"] = self.__performance_evaluator.calculate_crr(testing_labels, predicted_labels["L1"])
        crr["L2"] = self.__performance_evaluator.calculate_crr(testing_labels, predicted_labels["L2"])
        crr["COSINE"] = self.__performance_evaluator.calculate_crr(testing_labels, predicted_labels["COSINE"])

        return crr
    
    def evaluate_fmr_fnmr(self, verification_results):
        """
        Evaluates the False Match Rate (FMR) and False Non-Match Rate (FNMR) for the "COSINE" metric.
        
        Parameters:
            verification_results (dict): A dictionary containing the verification results for cosine distance. 
                The key is "COSINE", and the values are dictionaries containing the verification results for each threshold.

        Returns:
            tuple: A tuple containing two dictionaries:
                - fmr (dict): A dictionary where keys are metrics (e.g., "COSINE") and values are dictionaries 
                    mapping thresholds to their corresponding FMR values.
                - fnmr (dict): A dictionary where keys are metrics (e.g., "COSINE") and values are dictionaries 
                    mapping thresholds to their corresponding FNMR values.
        """
        fmr = {}
        fmr["COSINE"] = {}
        
        fnmr = {}
        fnmr["COSINE"] = {}
        
        for threshold in self.__thresholds:
            fmr["COSINE"][threshold] = self.__performance_evaluator.calculate_fmr(verification_results["COSINE"][threshold])
            fnmr["COSINE"][threshold] = self.__performance_evaluator.calculate_fnmr(verification_results["COSINE"][threshold])
        return fmr, fnmr
    
    def plot_det_curve(self, fmr, fnmr):
        """
        Plots the ROC curve for the given False Match Rate (FMR) and False Non-Match Rate (FNMR) values.

        Parameters:
            fmr (dict): A dictionary containing FMR values for different thresholds. The keys are the distance metric (e.g., "COSINE") 
                and the value is another dictionary with thresholds as keys and FMR values as values.
            fnmr (dict): A dictionary containing FNMR values for different thresholds. The keys are the distance metric (e.g., "COSINE") 
                and the value is another dictionary with thresholds as keys and FNMR values as values.
        """
        # Create arrays of FMR and FNMR values for different thresholds
        fmr_values = np.array([fmr["COSINE"][t] for t in self.__thresholds])
        fnmr_values = np.array([fnmr["COSINE"][t] for t in self.__thresholds])
        
        plt.figure(figsize=(10, 6))
        plt.plot(fmr_values, fnmr_values, marker="o", linestyle="-", color="r")

        # Annotate each point with its corresponding threshold
        for i, threshold in enumerate(self.__thresholds):
            plt.annotate(f"{threshold}", (fmr_values[i], fnmr_values[i]),
                     textcoords="offset points", xytext=(5, 5), ha="center", fontsize=8)
            
        plt.xlabel("False Match Rate (FMR) (in %)")
        plt.ylabel("False Non-Match Rate (FNMR) (in %)")
        plt.title("FMR vs. FNMR by different thresholds using COSINE distance")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.show()

def main():
    # Tunable parameters
    kernel_size = 31
    f = 0.075
    thresholds = [0.140, 0.150, 0.155, 0.160, 0.165, 0.170, 0.180, 0.190, 0.200, 0.300, 0.400]

    # Load the training and testing data
    training, testing = DataLoader.create().load()

    # Create the Iris Recognition Model using the tuned parameters
    iris_model = IrisRecognitionModel(training, testing, kernel_size, f, thresholds)

    # Extract features and labels from the training and testing data
    X_train, X_test, y_train, y_test = iris_model.extract_features_and_labels()

    # Fit the model using the training data
    iris_model.fit(X_train, y_train)

    # Identification mode: Identify the class labels for the testing data
    y_pred = iris_model.identify(X_test)

    # Verification mode: Verify the testing data against the true labels
    y_verif = iris_model.verify(X_test, y_test)

    # Evaluate the Correct Recognition Rate (CRR) and False Match Rate (FMR) and False Non-Match Rate (FNMR)
    crr = iris_model.evaluate_crr(y_test, y_pred)
    fmr, fnmr = iris_model.evaluate_fmr_fnmr(y_verif)

    # Print the results
    print(f"{'L1 distance measure':<25} | {round(crr['L1'], 4):>6} %")
    print(f"{'L2 distance measure':<25} | {round(crr['L2'], 4):>6} %")
    print(f"{'Cosine distance measure':<25} | {round(crr['COSINE'], 4):>6} %")

    fmr_rounded = {k: round(v, 2) for k, v in fmr['COSINE'].items()}
    print("Cosine FMR |")
    for threshold, rate in fmr_rounded.items():
        print(f"threshold {threshold:<5}: {rate:>6} %")
    
    fnmr_rounded = {k: round(v, 2) for k, v in fnmr['COSINE'].items()}
    print("Cosine FNMR |")
    for threshold, rate in fnmr_rounded.items():
        print(f"threshold {threshold:<5}: {rate:>6} %")

    # Plot the ROC curve
    iris_model.plot_det_curve(fmr, fnmr)

if __name__ == "__main__":
    main()