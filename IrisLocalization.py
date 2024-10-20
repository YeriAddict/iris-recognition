import cv2
import numpy as np

# TODO: Comment the code and improve code for detect_pupil, detect_sclera

class IrisLocalizer:
    def __init__(self, image):
        self.image = image
        self.region_dimension = 60
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        self.Xp = None
        self.Yp = None
        self.Rp = None

        self.Xs = None
        self.Ys = None
        self.Rs = None

    def estimate_pupil_center_coordinates(self):
        horizontal_projection = np.sum(self.image, axis=0)
        vertical_projection = np.sum(self.image, axis=1)

        self.Xp = np.argmin(horizontal_projection)
        self.Yp = np.argmin(vertical_projection)

    def refine_pupil_center_coordinates(self):
        # Define the region coordinates
        x1 = max(0, self.Xp - self.region_dimension)
        x2 = min(self.width, self.Xp + self.region_dimension)
        y1 = max(0, self.Yp - self.region_dimension)
        y2 = min(self.height, self.Yp + self.region_dimension)

        # Extract the pupil region based on the estimated center
        pupil_region = self.image[y1:y2, x1:x2]

        # Convert to binary image
        if len(pupil_region.shape) == 3 and pupil_region.shape[2] == 3:
            pupil_region = cv2.cvtColor(pupil_region, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding using Otsu's method
        _, threshold = cv2.threshold(pupil_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find the centroid of the binary region
        moments = cv2.moments(threshold)
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])

        self.Xp = centroid_x + x1
        self.Yp = centroid_y + y1
        self.Rp = np.sqrt(np.sum(threshold == 255) / np.pi)
   
    def detect_pupil(self):
        x1 = max(0, self.Xp - self.region_dimension)
        x2 = min(self.width, self.Xp + self.region_dimension)
        y1 = max(0, self.Yp - self.region_dimension)
        y2 = min(self.height, self.Yp + self.region_dimension)        

        # Extract the region of interest (ROI)
        pupil_region = self.image[y1:y2, x1:x2]

        # Check if the ROI has 3 channels (BGR), otherwise use as grayscale
        if len(pupil_region.shape) == 3 and pupil_region.shape[2] == 3:
            pupil_region = cv2.cvtColor(pupil_region, cv2.COLOR_BGR2GRAY)

        # Apply the Canny edge detector
        edges = cv2.Canny(pupil_region, 50, 150)

        # Apply the Hough Circle Transform to detect pupil circles
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=30, minRadius=20, maxRadius=50)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Draw the detected circle on the original image
                cv2.circle(self.image, (x + x1, y + y1), r, (235, 206, 135), 1)
                self.Rp = r  # Set the pupil radius
                return x + x1, y + y1, r
        return None

    def detect_sclera(self):
        # Define a region larger than the pupil region to cover the iris
        iris_region_dimension = int(self.Rp * 2.5)  # Adjust based on expected iris size

        # Ensure that the region doesn't go out of bounds
        x1 = max(0, self.Xp - iris_region_dimension)
        x2 = min(self.width, self.Xp + iris_region_dimension)
        y1 = max(0, self.Yp - iris_region_dimension)
        y2 = min(self.height, self.Yp + iris_region_dimension)        

        # Extract the region of interest (ROI) around the pupil for iris/sclera detection
        iris_region = self.image[y1:y2, x1:x2]

        # Check if the ROI has 3 channels (BGR), convert to grayscale if necessary
        if len(iris_region.shape) == 3 and iris_region.shape[2] == 3:
            iris_region = cv2.cvtColor(iris_region, cv2.COLOR_BGR2GRAY)

        # Apply the Canny edge detector to detect edges in the iris region
        edges = cv2.Canny(iris_region, 50, 150)

        # Apply the Hough Circle Transform to detect the iris circle (larger than the pupil)
        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=30, minRadius=int(self.Rp * 1.5), maxRadius=int(self.Rp * 3)
        )

        # If circles are detected, choose the first circle found
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Align the detected circle center with the stored pupil center
                self.Xs = x + x1
                self.Ys = y + y1
                self.Rs = r
                
                # Draw the detected iris circle on the original image
                cv2.circle(self.image, (self.Xs, self.Ys), self.Rs, (0, 255, 0), 2)
                
                return self.Xs, self.Ys, self.Rs  # Return coordinates and radius of the iris (approximating the sclera boundary)
        
        # If no circles are detected, return None
        return None
    
    def localize(self):
        self.estimate_pupil_center_coordinates()
        self.refine_pupil_center_coordinates()
        self.detect_pupil()
        self.detect_sclera()
        
        return (self.Xp, self.Yp, self.Rp), (self.Xs, self.Ys, self.Rs)
