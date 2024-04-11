# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: EyeballDetector.py
# Description: The EyeballDetector class is designed to accurately detect and estimate
#              the center and radius of an eyeball's sphere using a set of three-dimensional
#              points. It utilizes a confidence-based approach to continually refine these
#              estimations. The detector initializes with default parameters, including
#              assumptions about the eye's initial center and radius, and adjusts these
#              based on incoming data points. It incorporates a dynamic update mechanism
#              that relies on a minimum confidence threshold to initially detect the center
#              and continues to refine its estimations until a higher, reasonable confidence
#              level is achieved or the update period exceeds a specified threshold. This
#              class is particularly useful in applications requiring precise eye tracking
#              and analysis. The implementation is designed to be flexible, allowing
#              customization of key parameters to suit different accuracy and performance needs.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------


import numpy as np
from scipy.optimize import minimize
import time

class EyeballDetector:
    def __init__(self, initial_eye_center,
                 initial_eye_radius=0.02,
                 min_confidence=0.995,
                 reasonable_confidence=0.997,
                 points_threshold=300,
                 points_history_size=400,
                 refresh_time_threshold=10000):
        """
        Initializes the eyeball detector with customizable parameters for detecting the eye's sphere.

        Args:
        - initial_eye_center (np.array): Initial assumption about the eye's center.
        - initial_eye_radius (float): Initial assumption about the eye's radius.
        - min_confidence (float): Minimum confidence to consider the center detected.
        - reasonable_confidence (float): Confidence threshold to stop updating the center and radius.
        - points_threshold (int): Number of points required to start estimation.
        - points_history_size (int): Maximum size of the queue of collected points for calculating.
        - refresh_time_threshold (int): Time in milliseconds to refresh the detection state.
        """
        self.eye_center = np.array(initial_eye_center)
        self.eye_radius = initial_eye_radius
        self.min_confidence = min_confidence
        self.reasonable_confidence = reasonable_confidence
        self.points_threshold = points_threshold
        self.points_history_size = points_history_size
        self.refresh_time_threshold = refresh_time_threshold
        self.points_for_eye_center = None
        self.current_confidence = 0.0
        self.center_detected = False
        self.search_completed = False
        self.last_update_time = int(time.time() * 1000)

    def update(self, new_points, timestamp_ms):
        """
        Updates the detection of the eye's sphere center and radius based on current points and confidence.

        Args:
        - new_points (np.array): New points to add to detection set.
        - timestamp_ms (int): The current frame's timestamp in milliseconds.
        """

        if self.points_for_eye_center is not None:
            self.points_for_eye_center = np.concatenate((self.points_for_eye_center, new_points), axis=0)[-self.points_history_size:]
        else:
            self.points_for_eye_center = new_points

        if len(self.points_for_eye_center) >= self.points_threshold and not self.search_completed:
            center, radius, confidence = self._solve_for_sphere(self.points_for_eye_center)

            if confidence and confidence > self.current_confidence:
                self.eye_center = center
                self.eye_radius = radius
                self.current_confidence = confidence
                self.last_update_time = timestamp_ms

                if confidence >= self.min_confidence:
                    self.center_detected = True
                if confidence >= self.reasonable_confidence:
                    self.search_completed = True  # Indicate that the search can be concluded

            # Reset detection if too much time has passed without an update
            if (timestamp_ms - self.last_update_time) > self.refresh_time_threshold:
                self.search_completed = False

    def _solve_for_sphere(self, points, radius_bounds=(0.015, 0.025)):
        """
        Solves for the sphere's center and radius given a set of points.

        Args:
        - points (np.array): Array of points.
        - radius_bounds (tuple): Bounds for the sphere's radius (min_radius, max_radius).

        Returns:
        - tuple: The center (x, y, z), radius of the sphere, and the confidence of the solution.
        """
        def objective(params, points):
            center, R = params[:3], params[3]
            residuals = np.linalg.norm(points - center, axis=1) - R
            return np.sum(residuals**2)

        bounds = [(None, None), (None, None), (None, None), radius_bounds]
        result = minimize(objective, np.append(self.eye_center, self.eye_radius), args=(points,), bounds=bounds)

        if radius_bounds[0] <= result.x[3] <= radius_bounds[1]:
            center = result.x[:3]
            radius = result.x[3]
            confidence = 1 / (1 + result.fun)  # Inverse of loss
            return center, radius, confidence
        else:
            return None, None, None

    def reset(self):
        """
        Resets the detector to initial values and states.
        """
        self.points_for_eye_center = None
        self.current_confidence = 0.0
        self.center_detected = False
        self.search_completed = False
        self.last_update_time = int(time.time() * 1000)
