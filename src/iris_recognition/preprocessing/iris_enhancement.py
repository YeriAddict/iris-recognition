from iris_recognition.imports import cv2, np

class IrisEnhancer:
    """
    Class for enhancing the normalized iris image in order to obtain a more well-distributed texture image

    Attributes
    ----------
        image : numpy.ndarray
           The normalized iris image.
        background_illumination : numpy.ndarray
            The background illumination image.
        M : int
            The height of the normalized image.
        N : int
            The width of the normalized image.
        block_size_hist : int
            The size of the blocks used for histogram equalization.

    Methods
    -------
        enhance_iris():
            Enhances the iris image by subtracting the background illumination and applying histogram equalization in 32x32 blocks.
        save_image(filename):
            Saves the image with the enhanced iris.
    """
    def __init__(self, image, background_illumination):
        self.image = image

        self.__background_illumination = background_illumination
        self.__M = image.shape[0]
        self.__N = image.shape[1]

        self.__block_size_hist = 32

    def enhance_iris(self):
        """
        Enhances the normalized iris image.
        This method performs the following steps:
        1. Subtracts the background illumination from the normalized image to compensate for varying lighting conditions.
        2. Divides the image into 32x32 blocks.
        3. Applies histogram equalization to each block to enhance the contrast.
        
        Returns:
            np.ndarray: The enhanced iris image.
        """
        # Subtract from the normalized image to compensate for a variety of lighting conditions
        illum_estimate_image = cv2.subtract(self.image, self.__background_illumination.astype(np.uint8))

        # Redefine the image as a block matrix of dimensions 32x32
        enhanced_image = np.zeros((self.__M, self.__N))

        # Loop through each 32x32 block to apply histogram equalization
        for i in range(0, self.__M, self.__block_size_hist):
            for j in range(0, self.__N, self.__block_size_hist):
                i_range = min(i + self.__block_size_hist, self.__M)
                j_range = min(j + self.__block_size_hist, self.__N)
                block = illum_estimate_image[i:i_range, j:j_range]
                enhanced_image[i:i_range, j:j_range] = cv2.equalizeHist(block.astype(np.uint8))
            
        self.image = enhanced_image

        return self.image

    def save_image(self, filename):
        """
        Saves the image with the enhanced iris to a specified file.

        Parameters:
            filename (str): The path and name of the file where the image will be saved.
        """
        # Save the image with the enhanced iris
        cv2.imwrite(filename, self.image)