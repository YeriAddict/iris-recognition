import cv2
import numpy as np

class FeatureExtractor:
    """
    Class for extracting features from a normalized iris image using Gabor filters.

    Attributes
    ----------
    normalized_image : numpy.ndarray
        The normalized iris image.
    ksize : int
        Size of the Gabor kernel.
    um_theta: int
        Number of orientations for the Gabor filters.
    block_size : int
        The size of the blocks for local feature extraction.
    
    Methods
    -------
    extract_features():
        Extracts features from the normalized iris image using Gabor filters and returns the feature vector.
    """

    def __init__(self, normalized_image, ksize=31, num_theta = 2, block_size=8):
        self.normalized_image = normalized_image
        self.ksize = ksize
        self.num_theta = num_theta
        self.block_size = block_size

    def extract_features(self):
        """
        Extracts features from the normalized iris image using Gabor filters.

        Returns:
        - feature_vector: A vector containing the extracted features.
        """
        # Step 1: Define spatial filters. Implement Gabor filters or define
        # custom spatial filters according to the description. Prepare filters
        # to target specific frequencies and orientations as needed for the iris texture.

        # crop the image to 48 x 512 since this is the ROI
        cropped_normalized_iris_image = self.normalized_image[:48, :]

        # Define Gabor filters
        filters = []

        # Source: https://www.geeksforgeeks.org/opencv-getgaborkernel-method/

        # Standard deviations for the Gaussian envelope
        # smaller sigma will capture finer details
        # 1 and 3 chosen initially to try to capture local and global features
        # sigmas = [1, 3]
        sigmas = [3]
        # Wavelength of the sinusoidal factor
        # shorter wavelenths captures finer details and longer wavelengths capture broader features
        # pi/4 and pi/2 are common values
        # lambdas = [np.pi / 4, np.pi / 2]
        lambdas = [np.pi / 4]
        # Spatial aspect ratio
        # 1 is circular and less than one is an ellipse along an axis.
        # gammas = [0.5, 1.0]
        gammas = [0.5]

        # Orientation
        for theta in np.linspace(0, np.pi, self.num_theta):
            for sigma in sigmas:
                for lamda in lambdas:
                    for gamma in gammas:
                        # - num_theta is set to 0 for circular symmetric filtering
                        kernel = cv2.getGaborKernel((self.ksize, self.ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                        # plt.imshow(kernel, cmap='gray')
                        # plt.title("Kernel with orientation = " + str(theta))
                        # plt.show()
                        filters.append(kernel)

        # Step 2: Apply filters to the region of interest (ROI). Extract the region of
        # interest (ROI) from the normalized iris image. Apply each filter to the ROI
        # and store the results.

        # Apply each filter to the normalize (or enhanced?) iris image and extract features
        feature_vector = []
        for kernel in filters:
            # Apply the Gabor filter to extract texture information
            filtered_img = cv2.filter2D(cropped_normalized_iris_image, cv2.CV_8UC3, kernel)

            # Step 3: Divide the ROI into smaller blocks. Divide the filtered images into
            # small blocks (e.g., 8x8 blocks). This can be done by slicing the array into subarrays.
            # Divide the filtered image into blocks

            for i in range(0, filtered_img.shape[0], self.block_size):
                for j in range(0, filtered_img.shape[1], self.block_size):
                    # Step 4: Extract features from each block. Calculate the mean and standard deviation
                    # of each block. These statistics will be the feature vectors for the iris.
                    block = filtered_img[i:i+self.block_size, j:j+self.block_size]
                    if block.shape[0] == self.block_size and block.shape[1] == self.block_size:
                        # Calculate the mean and standard deviation for each block
                        mean_val = np.mean(block)
                        std_dev = np.std(block)
                        # Step 5: Compile Feature Vector. Combine all features from all
                        # filters and all blocks into a single feature vector.
                        feature_vector.extend([mean_val, std_dev])

        return feature_vector
