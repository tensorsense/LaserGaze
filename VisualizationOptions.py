# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: VisualizationOptions.py
# Description: A class to store visualization settings for rendering gaze vectors
#              on video frames.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

class VisualizationOptions:
    """
    A class to store visualization settings for rendering gaze vectors on video frames.
    """

    def __init__(self, color=(0, 255, 0), line_thickness=4, length_coefficient=5.0):
        """
        Initializes the visualization options.

        Args:
        - color (tuple): RGB color for the gaze lines. Default is green.
        - line_thickness (int): Thickness of the gaze lines. Default is 4.
        - length_coefficient (float): Multiplier for the length of the gaze vector. Default is 5.0.
        """
        self.color = color
        self.line_thickness = line_thickness
        self.length_coefficient = length_coefficient