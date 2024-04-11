# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: AffineTransformer.py
# Description: This class is designed to calculate and apply affine transformations
#              between two sets of 3D points, typically used for mapping points from
#              one coordinate system to another. It supports scaling based on defined
#              horizontal and vertical points, estimating an affine transformation matrix,
#              and converting points between the two coordinate systems using the matrix.
#              It is especially useful in projects involving facial recognition,
#              augmented reality, or any application where precise spatial transformations
#              are needed between different 3D models.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

import numpy as np
import cv2

class AffineTransformer:
    """
    A class to calculate and manage affine transformations between two 3D point sets.
    This class allows for precise spatial alignment of 3D models based on calculated
    scale factors and transformation matrices, facilitating tasks such as facial
    landmark transformations or other applications requiring model alignment.
    """

    def __init__(self, m1_points, m2_points, m1_hor_points, m1_ver_points, m2_hor_points, m2_ver_points):
        """
        Initializes the transformer by calculating the scale factor and the affine transformation matrix.

        Args:
        - m1_points (np.array): Numpy array of the first set of 3D points.
        - m2_points (np.array): Numpy array of the second set of 3D points to which the first set is aligned.
        - m1_hor_points (np.array): Horizontal reference points from the first model used to calculate scaling.
        - m1_ver_points (np.array): Vertical reference points from the first model used to calculate scaling.
        - m2_hor_points (np.array): Horizontal reference points from the second model used to calculate scaling.
        - m2_ver_points (np.array): Vertical reference points from the second model used to calculate scaling.
        """
        self.scale_factor = self._get_scale_factor(
            np.array(m1_hor_points),
            np.array(m1_ver_points),
            np.array(m2_hor_points),
            np.array(m2_ver_points)
        )

        scaled_m2_points = m2_points * self.scale_factor

        retval, M, inliers = cv2.estimateAffine3D(m1_points, scaled_m2_points)
        if retval:
            self.success = True
            self.transform_matrix = M
        else:
            self.success = False
            self.transform_matrix = None

    def _get_scale_factor(self, m1_hor_points, m1_ver_points, m2_hor_points, m2_ver_points):
        """
        Calculates the scale factor between two sets of reference points (horizontal and vertical).

        Args:
        - m1_hor_points (np.array): Horizontal reference points from the first model.
        - m1_ver_points (np.array): Vertical reference points from the first model.
        - m2_hor_points (np.array): Horizontal reference points from the second model.
        - m2_ver_points (np.array): Vertical reference points from the second model.

        Returns:
        - float: The calculated uniform scale factor to apply.
        """
        m1_width = np.linalg.norm(m1_hor_points[0] - m1_hor_points[1])
        m1_height = np.linalg.norm(m1_ver_points[0] - m1_ver_points[1])
        m2_width = np.linalg.norm(m2_hor_points[0] - m2_hor_points[1])
        m2_height = np.linalg.norm(m2_ver_points[0] - m2_ver_points[1])
        scale_width = m1_width / m2_width
        scale_height = m1_height / m2_height
        return (scale_width + scale_height) / 2

    def to_m2(self, m1_point):
        """
        Transforms a point from the first model space to the second model space using the affine transformation matrix.

        Args:
        - m1_point (np.array): The point in the first model's coordinate space.

        Returns:
        - np.array or None: Transformed point in the second model's space if the transformation was successful; otherwise None.
        """
        if self.success:
            m1_point_homogeneous = np.append(m1_point, 1)  # Convert to homogeneous coordinates
            return np.dot(self.transform_matrix, m1_point_homogeneous) / self.scale_factor
        else:
            return None

    def to_m1(self, m2_point):
        """
        Transforms a point from the second model space back to the first model space using the inverse of the affine transformation matrix.

        Args:
        - m2_point (np.array): The point in the second model's coordinate space.

        Returns:
        - np.array or None: Transformed point back in the first model's space if the transformation was successful; otherwise None.
        """
        if self.success:
            affine_transform_4x4 = np.vstack([self.transform_matrix, [0, 0, 0, 1]])
            inverse_affine_transform = np.linalg.inv(affine_transform_4x4)
            m2_point_homogeneous = np.append(m2_point * self.scale_factor, 1)  # Convert to homogeneous coordinates
            m1_point_homogeneous = np.dot(inverse_affine_transform, m2_point_homogeneous)

            # Convert back to non-homogeneous coordinates
            return (m1_point_homogeneous[:3] / m1_point_homogeneous[3])
        else:
            return None
