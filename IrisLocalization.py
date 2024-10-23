import cv2
import numpy as np

class IrisLocalizer:
    """
    Class for localizing the iris in an eye image.
    
    Attributes
    ----------
        image : numpy.ndarray
            The input image containing the eye.
        region_dimension : int
            The dimension of the region around the estimated pupil center for refinement. Default is 60 (e.g 120x120 region)
        height : int
            The height of the input image.
        width : int
            The width of the input image.
        Xp : int or None
            The x-coordinate of the pupil center.
        Yp : int or None
            The y-coordinate of the pupil center.
        Rp : int or None
            The radius of the pupil.

    Methods
    -------
        localize_iris() :
            Localizes the iris by estimating and refining the pupil center coordinates and then detecting the iris.
            Returns the image with the detected iris boundaries and the coordinates of the iris center and radius.
        save_image(filename) :
            Saves the image with the detected iris boundaries drawn on it.
    """

    def __init__(self, image):
        self.image = image
        self.region_dimension = 60
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        self.Xp = None
        self.Yp = None
        self.Rp = None

    def localize_iris(self):
        """
        Localizes the iris in the given image.
        This method performs the following steps:
        1. Computes horizontal and vertical projections of the image to estimate the pupil center.
        2. Defines a region around the estimated center and refines the pupil center using adaptive thresholding and centroid calculation.
        3. Applies a bilateral filter and Canny edge detection to the grayscale image, then uses the Hough Circle Transform to detect the iris.
        4. Creates a mask to isolate the iris region and updates the image to keep only the iris.

        Returns:
            tuple: A tuple containing the processed image and the coordinates of the iris center and radius (self.image, (self.Xp, self.Yp, self.Rp)).
        """
        ### STEP 1 ###

        # Estimate the pupil center as the center of the image
        self.Xp = self.image.shape[1] // 2
        self.Yp = self.image.shape[0] // 2

        ### STEP 2 ###

        # Define the 120x120 region around the estimated center
        x1 = max(0, self.Xp - self.region_dimension)
        x2 = min(self.width, self.Xp + self.region_dimension)
        y1 = max(0, self.Yp - self.region_dimension)
        y2 = min(self.height, self.Yp + self.region_dimension)
        pupil_region = self.image[y1:y2, x1:x2]

        # Convert to grayscale
        if len(pupil_region.shape) == 3 and pupil_region.shape[2] == 3:
            pupil_region = cv2.cvtColor(pupil_region, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding using Otsu's method
        _, threshold = cv2.threshold(pupil_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find the centroid of the binary region
        moments = cv2.moments(threshold)
        if moments["m00"] != 0:
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
        else:
            centroid_x = self.Xp
            centroid_y = self.Yp

        # Update the pupil center coordinates
        self.Xp = centroid_x + x1
        self.Yp = centroid_y + y1

        ### STEP 3 ###

        # Convert to grayscale
        gray_image = self.image
        if len(gray_image.shape) == 3 and gray_image.shape[2] == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply a bilateral filter to smooth the image while preserving edges
        bilateral = cv2.bilateralFilter(gray_image, 15, 75, 75)

        # Create a binary mask by thresholding the filtered image
        masked = cv2.inRange(bilateral, 0, 75)

        # Apply the mask to the original filtered image to retain only the pixels within the threshold range
        masked_img = cv2.bitwise_and(bilateral, masked)

        # Detect edges using the Canny edge detector
        edges = cv2.Canny(masked_img, 100, 220)

         # Use the Hough Circle Transform to detect circular shapes in the edge-detected image
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 5, 100)

        # Find the circle closest to the estimated center by calculating the Euclidean distance
        closest_circle = min(circles[0], key=lambda x: np.linalg.norm(np.array([self.Xp, self.Yp]) - np.array([x[0], x[1]])))

        # Update the pupil center coordinates
        self.Xp = int(closest_circle[0])
        self.Yp = int(closest_circle[1])
        self.Rp = int(closest_circle[2])
    
        ### STEP 4 ###

        # Create a mask to keep only the iris region
        mask = np.zeros_like(self.image)

        # Draw a white-filled circle over the iris area on the mask
        cv2.circle(mask, (self.Xp, self.Yp), int(self.Rp+55), (255, 255, 255), thickness=-1)

        # Keep only the iris region by performing a bitwise AND with the mask
        self.image = cv2.bitwise_and(self.image, mask)

        return self.image, (self.Xp, self.Yp, self.Rp)
    
    def save_image(self, filename):
        """
        Saves the image with the detected iris boundaries to a specified file.

        Parameters:
            filename (str): The path and name of the file where the image will be saved.
        """
        # Save the image with the detected iris boundaries
        cv2.imwrite(filename, self.image)
