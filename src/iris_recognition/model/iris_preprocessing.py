from iris_recognition.imports import os, json, IrisLocalizer, IrisNormalizer, IrisIlluminater, IrisEnhancer, FeatureExtractor

class IrisDataPreprocessor:
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

        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        self.__input_path = config["paths"]["input"]
        self.__output_path = config["paths"]["output"]
        self.__localized_images_path = config["paths"]["localized_folder"]
        self.__normalized_images_path = config["paths"]["normalized_folder"]
        self.__illuminated_images_path = config["paths"]["illuminated_folder"]
        self.__enhanced_images_path = config["paths"]["enhanced_folder"]

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
            label = os.path.normpath(original_image_path).split(os.sep)[2]

            # Store the features and labels
            self.features_vectors += features
            if mode == "Train":
                for _ in range(len(rotation_angles)):
                    self.labels.append(label)
            elif mode == "Test":
                self.labels.append(label)
            else:
                raise ValueError("Invalid mode")