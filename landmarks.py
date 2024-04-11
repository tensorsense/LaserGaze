# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: landmarks.py
# Description: This file defines a series of key facial landmark indices used for gaze
#              estimation and other facial analysis tasks within the LaserGaze project.
#              It includes definitions for the positions of outer head points, nose
#              bridge, nose tip, irises, pupils, and both internal and external eye corners.
#              Additionally, adjacent eyelid parts are defined for more detailed eye
#              tracking. Utility functions for converting normalized landmark coordinates
#              to pixel coordinates on images are also provided. These landmarks and
#              utilities facilitate precise tracking and manipulation of facial features
#              in real-time video processing.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

OUTER_HEAD_POINTS = [162, 389]
NOSE_BRIDGE = 6
NOSE_TIP = 4

LEFT_IRIS = [469, 470, 471, 472]
LEFT_PUPIL = 468

RIGHT_IRIS = [474, 475, 476, 477]
RIGHT_PUPIL = 473

INTERNAL_EYES_CORNERS = [155, 362]
OUTER_EYES_CORNERS = [33, 263]

ADJACENT_LEFT_EYELID_PART = [160, 159, 158, 163, 144, 145, 153]
ADJACENT_RIGHT_EYELID_PART = [387, 386, 385, 390, 373, 374, 380]

BASE_LANDMARKS = INTERNAL_EYES_CORNERS + OUTER_EYES_CORNERS + OUTER_HEAD_POINTS + [NOSE_BRIDGE] + [NOSE_TIP]

relative = lambda landmark, shape: (int(landmark[0] * shape[1]), int(landmark[1] * shape[0]))
relativeT = lambda landmark, shape: (int(landmark[0] * shape[1]), int(landmark[1] * shape[0]), 0)