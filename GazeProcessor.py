# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: GazeProcessor.py
# Description: This class processes video input to detect facial landmarks and estimate
#              gaze vectors using MediaPipe. The gaze estimation results are asynchronously
#              output via a callback function. This class leverages advanced facial
#              recognition and affine transformation to map detected landmarks into a
#              3D model space, enabling precise gaze vector calculation.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

import mediapipe as mp
import cv2
import time
from landmarks import *
from face_model import *
from AffineTransformer import AffineTransformer
from EyeballDetector import EyeballDetector

# Can be downloaded from https://developers.google.com/mediapipe/solutions/vision/face_landmarker
model_path = "./face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class GazeProcessor:
    """
    Processes video input to detect facial landmarks and estimate gaze vectors using the MediaPipe library.
    Outputs gaze vector estimates asynchronously via a provided callback function.
    """

    def __init__(self, camera_idx=0, callback=None, visualization_options=None):
        """
        Initializes the gaze processor with optional camera settings, callback, and visualization configurations.

        Args:
        - camera_idx (int): Index of the camera to be used for video capture.
        - callback (function): Asynchronous callback function to output the gaze vectors.
        - visualization_options (object): Options for visual feedback on the video frame. Supports visualization options
        for calibration and tracking states.
        """
        self.camera_idx = camera_idx
        self.callback = callback
        self.vis_options = visualization_options
        self.left_detector = EyeballDetector(DEFAULT_LEFT_EYE_CENTER_MODEL)
        self.right_detector = EyeballDetector(DEFAULT_RIGHT_EYE_CENTER_MODEL)
        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO
        )

    async def start(self):
        """
        Starts the video processing loop to detect facial landmarks and calculate gaze vectors.
        Continuously updates the video display and invokes callback with gaze data.
        """
        with FaceLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(self.camera_idx)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                timestamp_ms = int(time.time() * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                face_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if face_landmarker_result.face_landmarks:
                    lms_s = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarker_result.face_landmarks[0]])
                    lms_2 = (lms_s[:, :2] * [frame.shape[1], frame.shape[0]]).round().astype(int)

                    mp_hor_pts = [lms_s[i] for i in OUTER_HEAD_POINTS]
                    mp_ver_pts = [lms_s[i] for i in [NOSE_BRIDGE, NOSE_TIP]]
                    model_hor_pts = OUTER_HEAD_POINTS_MODEL
                    model_ver_pts = [NOSE_BRIDGE_MODEL, NOSE_TIP_MODEL]

                    at = AffineTransformer(lms_s[BASE_LANDMARKS,:], BASE_FACE_MODEL, mp_hor_pts, mp_ver_pts, model_hor_pts, model_ver_pts)

                    indices_for_left_eye_center_detection = LEFT_IRIS + ADJACENT_LEFT_EYELID_PART
                    left_eye_iris_points = lms_s[indices_for_left_eye_center_detection, :]
                    left_eye_iris_points_in_model_space = [at.to_m2(mpp) for mpp in left_eye_iris_points]
                    self.left_detector.update(left_eye_iris_points_in_model_space, timestamp_ms)

                    indices_for_right_eye_center_detection = RIGHT_IRIS + ADJACENT_RIGHT_EYELID_PART
                    right_eye_iris_points = lms_s[indices_for_right_eye_center_detection, :]
                    right_eye_iris_points_in_model_space = [at.to_m2(mpp) for mpp in right_eye_iris_points]
                    self.right_detector.update(right_eye_iris_points_in_model_space, timestamp_ms)

                    left_gaze_vector, right_gaze_vector = None, None

                    if self.left_detector.center_detected: #and self.right_detector.center_detected
                        left_eyeball_center = at.to_m1(self.left_detector.eye_center)
                        left_pupil = lms_s[LEFT_PUPIL]
                        left_gaze_vector = left_pupil - left_eyeball_center
                        left_proj_point = left_pupil + left_gaze_vector*5.0

                    if self.right_detector.center_detected:
                        right_eyeball_center = at.to_m1(self.right_detector.eye_center)
                        right_pupil = lms_s[RIGHT_PUPIL]
                        right_gaze_vector = right_pupil - right_eyeball_center
                        right_proj_point = right_pupil + right_gaze_vector*5.0

                    if self.callback and (left_gaze_vector is not None or right_gaze_vector is not None):
                        await self.callback(left_gaze_vector, right_gaze_vector)

                    if self.vis_options:
                        if self.left_detector.center_detected and self.right_detector.center_detected:
                            p1 = relative(left_pupil[:2], frame.shape)
                            p2 = relative(left_proj_point[:2], frame.shape)
                            frame = cv2.line(frame, p1, p2, self.vis_options.color, self.vis_options.line_thickness)
                            p1 = relative(right_pupil[:2], frame.shape)
                            p2 = relative(right_proj_point[:2], frame.shape)
                            frame = cv2.line(frame, p1, p2, self.vis_options.color, self.vis_options.line_thickness)
                        else:
                            text_location = (10, frame.shape[0] - 10)
                            cv2.putText(frame, "Calibration...", text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.vis_options.color, 2)


                cv2.imshow('LaserGaze', frame)
                if cv2.waitKey(5) & 0xFF == 27:
                    break