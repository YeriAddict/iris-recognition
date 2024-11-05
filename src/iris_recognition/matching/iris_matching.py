from iris_recognition.imports import np, LinearDiscriminantAnalysis

class IrisMatcher:
    """
    Class for performing iris matching under identification or verification mode.
    It matches an input feature vector to a class using the nearest center classifier.

    Attributes
    ----------
        lda : LinearDiscriminantAnalysis
            The LDA model used used to calculate the projection matrix W and for dimensionality reduction.
        class_centers : dict
            A dictionary storing the class centers in the reduced space.

    Methods
    -------
        calculate_distance(f, center, metric):
            Calculates the distance between a feature vector and a class center (L1, L2 or Cosine).
        fit(train_features, train_labels):
            Trains the LDA model.
        match(feature_vector, metric, claimed_class=None):
            Matches a vector to a class center using the specified metric.
    """
    
    def __init__(self, num_classes):
        self.class_centers = {}

        self.__lda = LinearDiscriminantAnalysis(n_components=min(num_classes - 1, 1536), solver="eigen", shrinkage="auto")

    def __calculate_distance(self, f, center, metric):
        """
        Calculates the distance between two feature vectors using the specified metric.
        
        Parameters:
            f (numpy.ndarray): The feature vector.
            center (numpy.ndarray): The center vector to compare against.
            metric (str): The distance metric to use. Options are "L1", "L2", and "COSINE".
        
        Returns:
            float: The calculated distance.
        
        Raises:
            ValueError: If an invalid metric is provided.
        """
        if metric == "L1":
            return np.sum(np.abs(f - center))
        elif metric == "L2":
            return np.sqrt(np.sum((f - center) ** 2))
        elif metric == "COSINE":
            return 1 - np.dot(f, center) / (np.linalg.norm(f) * np.linalg.norm(center))
        else:
            raise ValueError("Invalid metric")

    def fit(self, train_features, train_labels):
        """
        Fits the LDA model and computes the projection matrix W and class centers using the provided training features and labels.
        
        Parameters:
        -----------
            train_features (list): A list of feature vectors to train the model.
            train_labels (list): A list of corresponding labels for the feature vectors.
        """
        # Convert train_features to a numpy array
        train_features = np.vstack(train_features)

        # Convert train_labels to a numpy array
        train_labels = np.array(train_labels)

        # Fit the LDA model and reduce the dimensionality of the training features
        reduced_features = self.__lda.fit_transform(train_features, train_labels)

        # Compute class centers in the reduced space
        for label in np.unique(train_labels):
            # Get the features corresponding to the current class
            class_reduced_features = reduced_features[train_labels == label]

            # Calculate the mean vector for this class in reduced space
            self.class_centers[label] = np.mean(class_reduced_features, axis=0)

    def match(self, feature_vector, metric, claimed_class=None):
        """
        Projects the input feature vector to the reduced space and matches it to the closest class center in identification or verification mode
        
        Parameters:
            feature_vector (list): The feature vector to be matched.
            metric (str): The distance metric (L1, L2 or Cosine).
            claimed_class (optional, str): The class label to be verified against in verification mode.
        
        Returns:
            float (Verification mode): The distance to the claimed class center.
            tuple (Identification mode): The label of the closest class center and the distance to it.
        """
        # Project the feature vector into the reduced space using the existing projection matrix
        f = self.__lda.transform([feature_vector])[0]

        ### Verification mode (One-to-One Matching) ### 
        if claimed_class is not None:
            # Calculate the distance between the projected feature vector and the claimed class center
            center = self.class_centers[claimed_class]
            distance = self.__calculate_distance(f, center, metric)

            return distance

        ### Identification mode (One-to-Many Matching) ### 
        else:
            best_label = None
            best_distance = float('inf')

            # Find the closest class center to the feature vector
            for label, center in self.class_centers.items():
                distance = self.__calculate_distance(f, center, metric)

                # Update the best label and distance if a closer class center is found
                if distance < best_distance:
                    best_distance = distance
                    best_label = label

            return best_label, best_distance