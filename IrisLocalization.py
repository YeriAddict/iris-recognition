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
    estimate_pupil_center_coordinates():
        Estimates the initial coordinates of the pupil center using horizontal and vertical projections.
    refine_pupil_center_coordinates():
        Refines the estimated coordinates of the pupil center by binarizing a restricted region and by using adaptive thresholding and calculating the centroid with the moments.
    detect_iris():
        Detects the iris using edge detection and the Hough Circle Transform.
    localize():
        Localizes the iris by estimating and refining the pupil center coordinates and then detecting the iris.
    save_image(filename):
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

    def estimate_pupil_center_coordinates(self):
        # Compute horizontal and vertical projections
        horizontal_projection = np.sum(self.image, axis=0)
        vertical_projection = np.sum(self.image, axis=1)

        # Estimate the pupil center as the minimum of the projections
        self.Xp = np.argmin(horizontal_projection)
        self.Yp = np.argmin(vertical_projection)

    def refine_pupil_center_coordinates(self):
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
   
    def detect_iris(self):
        # Convert to grayscale
        gray_image = self.image
        if len(gray_image.shape) == 3 and gray_image.shape[2] == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Apply a bilateral filter to smooth the image while preserving edges
        bilateral = cv2.bilateralFilter(gray_image, 9, 75, 75)

        # Create a binary mask by thresholding the filtered image
        masked = cv2.inRange(bilateral, 0, 70)

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
    
    def localize(self):
        # Find the coordinates of the pupil center and the radius of the pupil
        self.estimate_pupil_center_coordinates()
        self.refine_pupil_center_coordinates()
        self.detect_iris()
        
        return (self.Xp, self.Yp, self.Rp)
    
    def save_image(self, filename):
        # Draw the inner boundary (pupil)
        cv2.circle(self.image, (self.Xp, self.Yp), self.Rp, (255, 255, 255), 1)

        # Draw the outer boundary (sclera)
        cv2.circle(self.image, (self.Xp, self.Yp), self.Rp + 55, (255, 255, 255), 1)

        # Save the image with the detected iris boundaries
        cv2.imwrite(filename, self.image)
