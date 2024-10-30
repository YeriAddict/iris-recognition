import cv2
import numpy as np

class FeatureExtractor:
    """
    Class for extracting features from an enhanced iris image using custom Gabor filters.

    Attributes
    ----------
        image : numpy.ndarray
            The enhanced iris image.
        roi_height : int
            The height of the region of interest (ROI) to be extracted from the image.
        roi_width : int
            The width of the region of interest (ROI) to be extracted from the image.
        rotation_angles : list
            A list of rotation angles to be used to rotate images.
        kernel_size : int
            The size of the custom Gabor kernel.
        f : float
            The frequency parameter for the custom Gabor kernel.
        block_size : int
            The size of the blocks used for feature extraction.
        delta_x1 : float
            The delta_x parameter for the first channel's Gabor kernel.
        delta_y1 : float
            The delta_y parameter for the first channel's Gabor kernel.
        delta_x2 : float
            The delta_x parameter for the second channel's Gabor kernel.
        delta_y2 : float
            The delta_y parameter for the second channel's Gabor kernel.
        features : list
            The extracted features vector from the iris image.

    Methods
    -------
        rotate_enhanced_image(image, angle):
            Rotates the given image by the specified angle
        kernel_function(x, y, f, delta_x, delta_y):
            Generates a Gabor kernel.
        apply_filter_to_roi(roi, delta_x, delta_y):
            Applies the Gabor filter to the region of interest (ROI).
        extract_features_from_roi(filtered_roi):
            Extracts features from the filtered ROI by dividing it into blocks and calculating the mean and average absolute deviation for each block.
        extract_features():
            Extracts features from the iris image by applying Gabor filters to the ROI and combining the features from both channels.
    """
    def __init__(self, image):
        self.image = image
        self.roi_height = 48
        self.roi_width = 512
        self.rotation_angles = [-9, -6, -3, 0, 3, 6, 9]

        # [BEST] 18, 0.08 with normalization
        self.kernel_size = 18   # can be tuned
        self.f = 0.08           # can be tuned
        self.block_size = 8

        # First Channel
        self.delta_x1 = 3
        self.delta_y1 = 1.5

        # Second Channel
        self.delta_x2 = 4.5
        self.delta_y2 = 1.5

        self.features = []

    def normalize_features(self, features):
        features = np.array(features)
        mean = features.mean()
        std = features.std() if features.std() != 0 else 1  # Avoid division by zero
        normalized_features = (features - mean) / std

        return normalized_features
    
    def rotate_enhanced_image(self, image, angle):
        """
        Rotates the given image by the specified angle.
        
        Parameters:
            image (numpy.ndarray): The enhanced image to be rotated.
            angle (float): The angle in degrees by which to rotate the image. Positive values mean counter-clockwise rotation.
        
        Returns:
            numpy.ndarray: The rotated image.
        """
        # Calculate the number of pixels to shift based on the angle
        width = image.shape[1]
        pixels_shift = (angle * width) / 360

        # Roll the image to the left or right based on the angle
        rotated_image = np.roll(image, int(pixels_shift), axis=1)

        return rotated_image

    def kernel_function(self, x, y, f, delta_x, delta_y):
        """
        Computes the Gabor kernel function for given parameters.

        Parameters:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            f (float): The frequency of the sinusoidal function.
            delta_x (float): The standard deviation of the Gaussian envelope along the x-axis.
            delta_y (float): The standard deviation of the Gaussian envelope along the y-axis.

        Returns:
            float: The value of the Gabor kernel at the given coordinates.
        """
        # Define the modulating function
        M1 = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))

        # Define the Gaussian envelope
        gabor_kernel = (1/(2 * np.pi * delta_x * delta_y)) * np.exp(-0.5 * (((x**2) / (delta_x**2)) + ((y**2) / (delta_y**2)))) * M1

        return gabor_kernel

    def apply_filter_to_roi(self, roi, delta_x, delta_y):
        """
        Applies the custom filter to the region of interest (ROI).

        Parameters:
            roi (numpy.ndarray): The region of interest in the image to which the filter will be applied.
            delta_x (float): The standard deviation of the Gaussian envelope along the x-axis.
            delta_y (float): The standard deviation of the Gaussian envelope along the y-axis.

        Returns:
            numpy.ndarray: The filtered region of interest.
        """
        # Source: https://www.geeksforgeeks.org/opencv-getgaborkernel-method/
        # Create a grid for generating the kernel
        center = self.kernel_size // 2
        x = np.arange(-center, center + 1)
        y = np.arange(-center, center + 1)
        X, Y = np.meshgrid(x, y)

        # Generate the kernel for a channel
        kernel = self.kernel_function(X, Y, self.f, delta_x, delta_y)

        # Apply the kernel to the ROI
        filtered_roi = cv2.filter2D(roi, -1, kernel)

        return filtered_roi

    def extract_features_from_roi(self, filtered_roi):
        """
        Extracts features from a filtered region of interest (ROI) in an image.

        Parameters:
            filtered_roi (numpy.ndarray): The filtered region of interest from which features are to be extracted.
        
        Returns:
            list: A list of features extracted from the ROI. Each block contributes two features (mean and average absolute deviation).
        """
        # Extract features from the filtered ROI
        features = []
        for i in range(0, self.roi_height, self.block_size):
            for j in range(0, self.roi_width, self.block_size):
                block = filtered_roi[i:i+self.block_size, j:j+self.block_size]
                block_mean = np.mean(np.abs(block))
                block_average_absolute_deviation = np.mean(np.abs(np.abs(block) - block_mean))
                feature = (block_mean, block_average_absolute_deviation)
                features.append(feature[0])
                features.append(feature[1])
        return features

    def extract_features(self):
        """
        Extracts features from the region of interest (ROI) of the enhanced iris image.
        This method performs the following steps:
        0. Rotates the enhanced image by different angles.
        1. Extracts the ROI from the enhanced iris image.
        2. Applies two different filters to the ROI.
        3. Extracts features from the filtered ROIs.
        4. Combines the features from both filtered ROIs.

        Returns:
            list: A list of tuples of the format: (features, angle) for a total of 7 angles.
        """
        
        for angle in self.rotation_angles:
            rotated_image = self.rotate_enhanced_image(self.image, angle)

            # Extract the region of interest (ROI) from the enhanced iris image
            roi_image = rotated_image[:self.roi_height, :self.roi_width]

            # Filter the ROI using the two channels
            filtered_roi1 = self.apply_filter_to_roi(roi_image, self.delta_x1, self.delta_y1)
            filtered_roi2 = self.apply_filter_to_roi(roi_image, self.delta_x2, self.delta_y2)

            # Determine the features for each channel
            filtered_roi1_features = self.extract_features_from_roi(filtered_roi1)
            filtered_roi2_features = self.extract_features_from_roi(filtered_roi2)

            # Combine the features from both channels
            features_vector = filtered_roi1_features + filtered_roi2_features
            # self.features.append(features_vector)

            # [ADDED] Normalize the feature
            normalized_features = self.normalize_features(features_vector)
            self.features.append(normalized_features)
            
            # features_and_angle = (features, angle)
            # self.features.append(features_and_angle)

        return self.features