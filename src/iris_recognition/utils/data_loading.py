from iris_recognition.imports import cv2, os, json

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

        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        self.__input_path = config["paths"]["input"]

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