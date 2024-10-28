import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class IrisMatcher:
    """
    Class for matching iris feature vectors to their respective classes using LDA and nearest center classifier.

    Attributes
    ----------
        lda : LinearDiscriminantAnalysis
            The LDA model for dimensionality reduction.
        class_centers : dict
            A dictionary mapping class labels to their feature vectors.
        rotation_angles : list
            List of angles for rotation invariance.
    Methods
    -------
        fit(train_features, train_labels):
            Fits the LDA model using training feature vectors and labels.
        match(f):
            Classifies an input feature vector f using the nearest center classifier.
    """
    
    def __init__(self, num_classes):
        self.lda = LinearDiscriminantAnalysis(n_components = min(num_classes - 1, 1536))
        self.class_centers = {}
        self.rotation_angles = [-9, -6, -3, 0, 3, 6, 9]

    def fit(self, train_features, train_labels):
        """
        Fits the LDA model using the training features and labels.
        Computes the class centers in the reduced-dimensional space.
        
        Parameters:
            train_features : array-like, shape (n_samples, 1536)
                The training feature vectors.
            train_labels : array-like, shape (n_samples,)
                The class labels for the training samples.
        """
        train_labels = np.array(train_labels)
        
        # Fit the LDA model to find the projection matrix W
        self.lda.fit(train_features, train_labels)
        
        # Project training data to reduced-dimensional space
        reduced_features = self.lda.transform(train_features)
        # print(f"Reduced feature dimensions: {reduced_features.shape}")  # Should be (324, 107)
        # print(f"Shape of train_labels: {train_labels.shape}")  # Should be (324,)

        # Ensure train_labels is 1D
        train_labels = train_labels.flatten()
        
        # Calculate class centers in the reduced space
        unique_labels = np.unique(train_labels)
        # print(f"Unique labels: {unique_labels}")  # List the unique labels

        for label in unique_labels:
            # Select the reduced features belonging to the current class
            class_reduced_features = reduced_features[train_labels == label]
            # print(f"Label {label}: {class_reduced_features.shape}")  # Should be (3, 107) for each class

            # Compute the mean vector for this class
            class_center = np.mean(class_reduced_features, axis=0)
            # print(f"Center for class {label}: {class_center.shape}")  # Should be (107,)

            self.class_centers[label] = class_center  # Each center should have shape (107,)

        # Verify the shape of class centers for debugging
        for label, center in self.class_centers.items():
            print(f"Shape of center for class {label}: {center.shape}")

    def match(self, feature_vector):
        """
        Matches an input feature vector to a class using the nearest center classifier.
        
        Parameters:
            feature_vector : array-like, shape (1536,)
                The input feature vector to classify.
        
        Returns:
            tuple: (predicted_label, (d1, d2, d3))
                The predicted class label and the distances (L1, L2, Cosine) to the closest class center.
        """
        # Project the feature vector into the reduced-dimensional space
        reduced_f = self.lda.transform([feature_vector])[0]

        # Print shapes for debugging
        print(f"Shape of reduced_f: {reduced_f.shape}")

        # Initialize variables to store the best match
        best_label = None
        best_distance = float('inf')
        best_d1, best_d2, best_d3 = None, None, None

        # Iterate over each class center
        for label, center in self.class_centers.items():
            # Ensure that the center has the same shape as the reduced feature
            print(f"Shape of center for class {label}: {center.shape}")

            min_distance = float('inf')

            # Iterate over each rotation angle for matching
            for angle in self.rotation_angles:
                # Adjust the feature vector based on rotation (if necessary)
                rotated_f = self.rotate_feature(reduced_f, angle)

                # # Check shapes before calculating distances
                # if rotated_f.shape != center.shape:
                #     print(f"Error: Shape mismatch between rotated_f {rotated_f.shape} and center {center.shape}")
                #     continue

                # Calculate distances
                d1 = np.sum(np.abs(rotated_f - center))  # L1 distance
                d2 = np.sum((rotated_f - center) ** 2)   # L2 distance
                d3 = 1 - np.dot(rotated_f, center) / (np.linalg.norm(rotated_f) * np.linalg.norm(center))  # Cosine similarity

                # Update the minimum distance for this class
                if d2 < min_distance:
                    min_distance = d2
                    current_d1, current_d2, current_d3 = d1, d2, d3

            # Check if this class is the best match so far
            if min_distance < best_distance:
                best_distance = min_distance
                best_label = label
                best_d1, best_d2, best_d3 = current_d1, current_d2, current_d3

        return best_label, (best_d1, best_d2, best_d3)

    def rotate_feature(self, feature, angle):
        """
        Simulates rotation of a feature vector to achieve rotation invariance.
        For simplicity, this function may just return the feature as-is or include adjustments as needed.

        Parameters:
            feature : array-like
                The feature vector to rotate.
            angle : float
                The angle by which to adjust the feature vector.
        
        Returns:
            array-like: Adjusted feature vector.
        """
        # In this simplified implementation, the rotation logic can be adjusted as necessary.
        # For now, we assume the feature vector is invariant to the angle.
        return feature
