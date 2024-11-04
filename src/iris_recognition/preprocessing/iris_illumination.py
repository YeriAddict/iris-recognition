from iris_recognition.imports import cv2, np

class IrisIlluminater:
    """
    Class for obtaining a coarse estimation of the background illumination in an iris image.

    Attributes
    ----------
        image : numpy.ndarray
            The normalized iris image.
        M : int
            The height of the normalized image.
        N : int
            The width of the normalized image.
        block_size_light : int
            The size of the blocks used for calculating the mean value.

    Methods
    -------
        illuminate_iris():
            Estimates the background illumination of the iris in the image by dividing the image into blocks, 
            calculating the mean value of each block, and resizing the block matrix to the original image size using bicubic interpolation.
        save_image(filename):
            Saves the image with the estimated background illumination.
    """
    def __init__(self, image):
        self.image = image

        self.__M = image.shape[0]
        self.__N = image.shape[1]

        self.__block_size_light = 16

    def illuminate_iris(self):
        """
        Estimates the background illumination of the iris.
        This method performs the following steps:
        1. Divides the image into 16x16 blocks.
        2. Calculates the mean value for each block.
        3. Constructs a block matrix of mean values.
        4. Resizes the block matrix to the original image size using bicubic interpolation.

        Returns:
            np.ndarray: The image of the background illumination.
        """

        # Redefine the image as a block matrix of dimensions 16x16
        image_as_blocks = np.zeros((self.__M//self.__block_size_light, self.__N//self.__block_size_light))

        # Loop through each 16x16 block to calculate the mean value
        for i in range(0, self.__M, self.__block_size_light):
            for j in range(0, self.__N, self.__block_size_light):
                i_step = i // self.__block_size_light
                j_step = j // self.__block_size_light
                block = self.image[i:i+self.__block_size_light, j:j+self.__block_size_light]
                image_as_blocks[i_step, j_step] = np.mean(block)
                
        # Expand at the same size as the normalized image by bicubic interpolation 
        # (estimating the color in an image pixel by calculating the average of 16 pixels residing around pixels that are similar to pixels in the source image)
        background_illumination = cv2.resize(image_as_blocks, (self.__N, self.__M), interpolation=cv2.INTER_CUBIC)
        
        self.image = background_illumination

        return self.image

    def save_image(self, filename):
        """
        Saves the image with the background illumination for an iris to a specified file.

        Parameters:
            filename (str): The path and name of the file where the image will be saved.
        """
        # Save the image with the background illumination for an iris
        cv2.imwrite(filename, self.image)