import cv2
import numpy as np

class IrisNormalizer:
    """
    Class for normalizing the iris image to a rectangular shape of identical dimensions.

    Attributes
    ----------
        image : numpy.ndarray
            The original iris image.
        Xp : int
            The x-coordinate of the pupil center.
        Yp : int
            The y-coordinate of the pupil center.
        Rp : int 
            The radius of the pupil.
        M : int
            The height of the normalized image.
        N : int
            The width of the normalized image.
    Methods:
        normalize_iris():
            Normalizes the iris image by remapping it to polar coordinates from cartesian coordinates.
        save_image(filename) :
            Saves the image with the normalized iris.
    """
    def __init__(self, image, pupil_coordinates):
        self.image = image
        self.Xp = pupil_coordinates[0]
        self.Yp = pupil_coordinates[1]
        self.Rp = pupil_coordinates[2]
        
        self.M = 64
        self.N = 512

    def normalize_iris(self):
        """
        Normalizes the iris region of the eye image.
        This method performs the following steps:
        1. Generates the radial and angular coordinates and grids of shape MxN.
        2. Computes the inner and outer boundaries of the iris.
        3. Remaps the original image to the normalized coordinates.

        Returns:
            numpy.ndarray: The normalized iris image.
        """
        # Generate the radial and angular coordinates and grids of shape MxN
        r = np.linspace(0, 1, self.M)
        theta = np.linspace(0, 2 * np.pi, self.N)
        theta_grid, r_grid = np.meshgrid(theta, r)

        # Inner boundary
        rp = self.Rp
        xp = rp * np.cos(theta_grid)
        yp = rp * np.sin(theta_grid)

        # Outer boundary
        ri = self.Rp + 55
        xi = ri * np.cos(theta_grid)
        yi = ri * np.sin(theta_grid)

        # Store new pixel positions for each point on the iris
        x = self.Xp + (1 - r_grid) * xp + r_grid * xi
        y = self.Yp + (1 - r_grid) * yp + r_grid * yi

        # Remap the original image to the normalized coordinates
        normalized_iris = cv2.remap(self.image, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

        self.image = normalized_iris

        return self.image

    def save_image(self, filename):
        """
        Saves the image with the normalized iris to a specified file.

        Parameters:
            filename (str): The path and name of the file where the image will be saved.
        """
        # Save the image with the normalized iris
        cv2.imwrite(filename, self.image)