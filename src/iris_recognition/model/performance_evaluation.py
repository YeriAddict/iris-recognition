class PerformanceEvaluator:
    """
    Class for evaluating the performance of our model.

    Attributes
    ----------
        n_classes : int
            The number of classes in the dataset.
    
    Methods
    -------
        calculate_crr(labels, predicted_labels):
            Calculates the Correct Recognition Rate (CRR).
        calculate_fmr(verification_results):
            Calculates the False Match Rate (FMR).
        calculate_fnmr(verification_results):
            Calculates the False Non-Match Rate (FNMR).        
    """
    def __init__(self, n_classes):
        self.__n_classes = n_classes

    def calculate_crr(self, labels, predicted_labels):
        """
        Calculates the Correct Recognition Rate (CRR): percentage of correctly recognized labels out of the total number of classes.

        Parameters:
            labels (list): The true labels.
            predicted_labels (list): The predicted labels.

        Returns:
            float: The correct recognition rate as a percentage.
        """
        correct_recognition_rate = (sum(1 for true_label, predict_label in zip(labels, predicted_labels) if true_label == predict_label) / self.__n_classes) * 100
        return correct_recognition_rate

    def calculate_fmr(self, verification_results):
        """
        Calculates the False Match Rate (FMR).
        The FMR is the percentage of impostor attempts that are incorrectly accepted as genuine matches.

        Parameters:
            verification_results (dict): A dictionary containing the verification results with the following keys:
                - "false_matches" (int): The number of impostor matches incorrectly classified as matches.
                - "total_impostor_matches" (int): The total number of impostor match attempts.
        
        Returns:
            float: The False Match Rate (FMR) as a percentage.
        """
        false_match_rate = (verification_results["false_matches"] / verification_results["total_impostor_matches"]) * 100
        return false_match_rate

    def calculate_fnmr(self, verification_results):
        """
        Calculates the False Non-Match Rate (FNMR).
        The FNMR is the percentage of genuine matches that are incorrectly classified as non-matches.
        
        Parameters:
            verification_results (dict): A dictionary containing the verification results with the following keys:
                - "false_non_matches" (int): The number of genuine matches incorrectly classified as non-matches.
                - "total_genuine_matches" (int): The total number of genuine match attempts.
        Returns:
            float: The False Non-Match Rate (FNMR) as a percentage.
        """
        false_non_match_rate = (verification_results["false_non_matches"] / verification_results["total_genuine_matches"]) * 100
        return false_non_match_rate
