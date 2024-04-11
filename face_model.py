# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: face_model.py
# Description: A universal face model used for creating a head coordinate space. This
#              module contains definitions of key facial points and is used for
#              calculations within the context of head movement recognition and
#              analysis.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

import numpy as np

INTERNAL_EYES_CORNERS_MODEL = np.array([
    [-0.035, -0.05, 0],
    [0.035, -0.05, 0]
])

OUTER_EYES_CORNERS_MODEL = np.array([
    [-0.09, -0.057, 0.01],
    [0.09, -0.057, 0.01]
])

OUTER_HEAD_POINTS_MODEL = np.array([
    [-0.145, -0.1, 0.1],
    [0.145, -0.1, 0.1]
])

NOSE_BRIDGE_MODEL = np.array([
    [0, -0.0319, -0.0432]
])

NOSE_TIP_MODEL = np.array([
    [0, 0.088, -0.071]
])

BASE_FACE_MODEL = np.vstack((
    INTERNAL_EYES_CORNERS_MODEL,
    OUTER_EYES_CORNERS_MODEL,
    OUTER_HEAD_POINTS_MODEL,
    NOSE_BRIDGE_MODEL,
    NOSE_TIP_MODEL
))

DEFAULT_LEFT_EYE_CENTER_MODEL = (np.array(INTERNAL_EYES_CORNERS_MODEL[0]) + np.array(OUTER_EYES_CORNERS_MODEL[0])) * 0.5
DEFAULT_LEFT_EYE_CENTER_MODEL[1] -= 0.009
DEFAULT_LEFT_EYE_CENTER_MODEL[2] = 0.02

DEFAULT_RIGHT_EYE_CENTER_MODEL = (np.array(INTERNAL_EYES_CORNERS_MODEL[1]) + np.array(OUTER_EYES_CORNERS_MODEL[1])) * 0.5
DEFAULT_RIGHT_EYE_CENTER_MODEL[1] -= 0.009
DEFAULT_RIGHT_EYE_CENTER_MODEL[2] = 0.02

DEFAULT_EYE_RADIUS = 0.02
