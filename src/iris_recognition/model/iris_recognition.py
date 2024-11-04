from iris_recognition.imports import np, random, plt, IrisMatcher, IrisDataPreprocessor, PerformanceEvaluator

class IrisRecognitionModel:
    """
    Class for the built Iris Recognition Model.
    It uses the IrisDataPreprocessor, IrisMatcher, and PerformanceEvaluator classes.

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
        training_iris_preprocessor = IrisDataPreprocessor(self.__training)
        training_iris_preprocessor.localize_irises()
        training_iris_preprocessor.normalize_irises()
        training_iris_preprocessor.illuminate_irises()
        training_iris_preprocessor.enhance_irises()
        training_iris_preprocessor.extract_irises_features(self.__rotation_angles, self.__kernel_size, self.__f, self.__training_mode)
        training_features = training_iris_preprocessor.features_vectors
        training_labels = training_iris_preprocessor.labels

        # Testing
        testing_iris_preprocessor = IrisDataPreprocessor(self.__testing)
        testing_iris_preprocessor.localize_irises()
        testing_iris_preprocessor.normalize_irises()
        testing_iris_preprocessor.illuminate_irises()
        testing_iris_preprocessor.enhance_irises()
        testing_iris_preprocessor.extract_irises_features(self.__rotation_angles, self.__kernel_size, self.__f, self.__testing_mode)
        testing_features = testing_iris_preprocessor.features_vectors
        testing_labels = testing_iris_preprocessor.labels

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