import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class IrisMatcher:
    """
    Class for matching iris feature vectors to their respective classes using Fisher Linear Discriminant (FLD)
    projection matrix W and nearest center classifier in reduced space.

    Attributes
    ----------
        lda : LinearDiscriminantAnalysis
            The LDA model used to calculate the projection matrix W.
        class_centers : dict
            Dictionary storing class centers in the reduced space.

    Methods
    -------
        fit(train_features, train_labels):
            Fits the LDA model and computes the projection matrix W and class centers.
        match(feature_vector):
            Projects the input feature vector to the reduced space and matches it to the closest class center.
    """
    
    def __init__(self, num_classes):
        self.lda = LinearDiscriminantAnalysis(n_components=min(num_classes - 1, 1536))
        self.class_centers = {}

    def fit(self, train_features, train_labels):
        """
        Fits the LDA model using the training features and labels.
        Computes the projection matrix W and class centers in the reduced space.
        
        Parameters:
            train_features : array-like, shape (n_samples, 1536)
                The training feature vectors.
            train_labels : array-like, shape (n_samples,)
                The class labels for the training samples.
        """
        # Convert train_features from shape (2268, 1) to (2268, 1536)
        train_features = np.vstack(train_features)
        train_labels = np.array(train_labels)
        reduced_features = self.lda.fit_transform(train_features, train_labels)

        # Compute class centers in the reduced space
        for label in np.unique(train_labels):
            # Select projected vectors (f) of the current class
            class_reduced_features = reduced_features[train_labels == label]
            # Calculate the mean vector for this class in reduced space
            self.class_centers[label] = np.mean(class_reduced_features, axis=0)

    def match(self, feature_vector, metric):
        """
        Matches an input feature vector to a class using the nearest center classifier.
        
        Parameters:
            feature_vector : array-like, shape (1536,)
                The input feature vector to classify.
            metric : string, either L1, L2, or COSINE
        
        Returns:
            tuple: (predicted_label, best_distance)
                The predicted class label and the corresponding (best) distance.
        """
        # Project the feature vector into the reduced space using the existing projection matrix
        f = self.lda.transform([feature_vector])[0]

        best_label = None
        best_distance = float('inf')

        # Measure distances (d1, d2, d3) to each class center
        for label, center in self.class_centers.items():
            if metric == "L1":
                distance = np.sum(np.abs(f - center)) 
            elif metric == "L2":
                distance = np.sum((f - center) ** 2) # np.sqrt(np.sum((f - center) ** 2))
            elif metric == "COSINE":
                distance = 1 - np.dot(f.T, center) / (np.linalg.norm(f) * np.linalg.norm(center))
            else:
                return print("WARN: Wrong input for metric")
            
            if distance < best_distance:
                best_distance = distance
                best_label = label

        return best_label, best_distance