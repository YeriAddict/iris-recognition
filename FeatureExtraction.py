import cv2
import numpy as np

class FeatureExtractor:
    """
    Class for extracting features from an enhanced iris image using custom Gabor filters.

    Attributes
    ----------
    
    Methods
    -------
    """

    def __init__(self, image):
        self.image = image
        self.roi_height = 48
        self.roi_width = 512

        self.kernel_size = 31
        self.f = 0.1

        # First Channel
        self.delta_x1 = 3
        self.delta_y1 = 1.5

        # Second Channel
        self.delta_x2 = 4.5
        self.delta_y2 = 1.5
        
    def kernel_function(self, x, y, f, delta_x, delta_y):
        M1 = np.cos(2 * np.pi * f * np.sqrt(x**2 + y**2))
        gabor_kernel = (1/(2 * np.pi * delta_x * delta_y)) * np.exp(-0.5 * ((x**2 / delta_x**2) + (y**2 / delta_y**2))) * M1
        return gabor_kernel

    def apply_filter_to_roi(self, roi, delta_x, delta_y):
        # Create a grid for generating the kernel
        center = self.kernel_size // 2
        x = np.arange(-center, center + 1)
        y = np.arange(-center, center + 1)
        X, Y = np.meshgrid(x, y)

        # Generate the kernel for a channel
        kernel = self.kernel_function(X, Y, self.f, delta_x, delta_y)

        # Apply the kernel to the ROI
        filtered_roi = cv2.filter2D(roi, cv2.CV_32F, kernel)

        return filtered_roi

    def extract_energy_features(self, filtered_roi):
        # Divide the filtered ROI into three subregions
        subregions = np.array_split(filtered_roi, 3, axis=0)

        # Calculate the energy of each subregion
        energies = []
        for subregion in subregions:
            energy = np.sum(subregion**2)
            energies.append(energy)

        return energies

    def extract_features(self):
        # Extract the region of interest (ROI) from the enhanced iris image
        roi_image = self.image[:self.roi_height, :self.roi_width]

        # Filter the ROI using the two channels
        filtered_roi1 = self.filter_roi(roi_image, self.delta_x1, self.delta_y1)
        filtered_roi2 = self.filter_roi(roi_image, self.delta_x2, self.delta_y2)

        # Calculate the energy of each subregion for each channel
        filtered_roi1_energies = self.calculate_roi_energies(filtered_roi1)
        filtered_roi2_energies = self.calculate_roi_energies(filtered_roi2)

        # Combine the energies from both channels
        roi_energies = filtered_roi1_energies + filtered_roi2_energies


    # def extract_features(self):
    #     """
    #     Extracts features from the normalized iris image using Gabor filters.

    #     Returns:
    #     - feature_vector: A vector containing the extracted features.
    #     """

    #     # Extract the region of interest (ROI) from the enhanced iris image
    #     roi_image = self.image[:self.roi_height, :self.roi_width]

    #     # Step 1: Define spatial filters. Implement Gabor filters or define
    #     # custom spatial filters according to the description. Prepare filters
    #     # to target specific frequencies and orientations as needed for the iris texture.

    #     # Define Gabor filters
    #     filters = []

    #     # Source: https://www.geeksforgeeks.org/opencv-getgaborkernel-method/

    #     # Standard deviations for the Gaussian envelope
    #     # smaller sigma will capture finer details
    #     # 1 and 3 chosen initially to try to capture local and global features
    #     # sigmas = [1, 3]
    #     sigmas = [3]
    #     # Wavelength of the sinusoidal factor
    #     # shorter wavelenths captures finer details and longer wavelengths capture broader features
    #     # pi/4 and pi/2 are common values
    #     # lambdas = [np.pi / 4, np.pi / 2]
    #     lambdas = [np.pi / 4]
    #     # Spatial aspect ratio
    #     # 1 is circular and less than one is an ellipse along an axis.
    #     # gammas = [0.5, 1.0]
    #     gammas = [0.5]

    #     # Orientation
    #     for theta in np.linspace(0, np.pi, self.num_theta):
    #         for sigma in sigmas:
    #             for lamda in lambdas:
    #                 for gamma in gammas:
    #                     # - num_theta is set to 0 for circular symmetric filtering
    #                     kernel = cv2.getGaborKernel((self.kernel_size, self.kernel_size), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
    #                     # plt.imshow(kernel, cmap='gray')
    #                     # plt.title("Kernel with orientation = " + str(theta))
    #                     # plt.show()
    #                     filters.append(kernel)

    #     # Step 2: Apply filters to the region of interest (ROI). Extract the region of
    #     # interest (ROI) from the normalized iris image. Apply each filter to the ROI
    #     # and store the results.

    #     # Apply each filter to the normalize (or enhanced?) iris image and extract features
    #     feature_vector = []
    #     for kernel in filters:
    #         # Apply the Gabor filter to extract texture information
    #         filtered_img = cv2.filter2D(roi_image, cv2.CV_8UC3, kernel)

    #         # Step 3: Divide the ROI into smaller blocks. Divide the filtered images into
    #         # small blocks (e.g., 8x8 blocks). This can be done by slicing the array into subarrays.
    #         # Divide the filtered image into blocks

    #         for i in range(0, filtered_img.shape[0], self.block_size):
    #             for j in range(0, filtered_img.shape[1], self.block_size):
    #                 # Step 4: Extract features from each block. Calculate the mean and standard deviation
    #                 # of each block. These statistics will be the feature vectors for the iris.
    #                 block = filtered_img[i:i+self.block_size, j:j+self.block_size]
    #                 if block.shape[0] == self.block_size and block.shape[1] == self.block_size:
    #                     # Calculate the mean and standard deviation for each block
    #                     mean_val = np.mean(block)
    #                     std_dev = np.std(block)
    #                     # Step 5: Compile Feature Vector. Combine all features from all
    #                     # filters and all blocks into a single feature vector.
    #                     feature_vector.extend([mean_val, std_dev])

    #     return feature_vector
