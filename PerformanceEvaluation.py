class PerformanceEvaluator:
    """
    Class for evaluating the performance of our model.

    Attributes
    ----------
        n_classes : int
            The number of classes in the dataset.
    
    Methods
    -------
        calculate_crr(labels, predicted_labels)
            Calculates the Correct Recognition Rate (CRR) given the true labels and predicted labels.
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def calculate_crr(self, labels, predicted_labels):
        """
        Calculates the Correct Recognition Rate (CRR): percentage of correctly recognized labels out of the total number of classes.

        Parameters:
            labels (list): The true labels.
            predicted_labels (list): The predicted labels.

        Returns:
            float: The correct recognition rate as a percentage.
        """
        correct_recognition_rate = (sum(1 for true_label, predict_label in zip(labels, predicted_labels) if true_label == predict_label) / self.n_classes) * 100
        return correct_recognition_rate

