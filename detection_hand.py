# # import the opencv library
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python

from draw_hand import draw_landmarks_on_image


# define a video capture object

class DetectionHand:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.result_now = self.HandLandmarkerResult(handedness=[], hand_landmarks=[], hand_world_landmarks=[])
        self.options = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path='../model/hand_landmarker.task'),
                                             running_mode=VisionRunningMode.LIVE_STREAM,
                                             result_callback=self.make_result)
        self.image_draw = np.full((1000, 1000, 3), 255, dtype=np.uint8)
        self.image_draw_height = 1000
        self.image_draw_width = 1000
        self.color = (0, 0, 255)
        self.radius = 4
        self.draw = False
        self.position = "Active"

    def check_draw_position(self, res):
        res = res[0]
        if (res[8].y > res[5].y and res[12].y > res[9].y and
                res[16].y > res[13].y and res[20].y > res[17].y):
            if self.position == "Active":
                self.position = "Pause"
                self.draw = not self.draw
        else:
            self.position = "Active"

    # cond_2 =
    # return middle_finger_tip_y - index_finger_tip_y > 0.30

    @staticmethod
    def check_change_color(hand_landmarks):
        index_finger_tip_y = hand_landmarks[0][8].y
        middle_finger_tip_y = hand_landmarks[0][12].y
        return index_finger_tip_y - middle_finger_tip_y > 0.30

    def start_draw(self, result: mp.tasks.vision.HandLandmarkerResult):
        draw_x = result.hand_landmarks[0][8].x
        draw_y = result.hand_landmarks[0][8].y
        draw_x_normal = draw_x * self.image_draw_width
        draw_y_normal = draw_y * self.image_draw_height
        self.image_draw = cv2.circle(self.image_draw, (int(draw_x_normal), int(draw_y_normal)), radius=self.radius,
                                     color=self.color, thickness=-1)
        # Create a hand landmarker instance with the live stream mode:

    def make_result(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """ print('---------------------------------hand landmarker result: {}'.format(result)) """
        if result.hand_landmarks:
            # print("index_finger_tip", result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y)
            # print("middle_finger_tip", result.hand_landmarks[0][12].x, result.hand_landmarks[0][12].y)
            # print("ring_finger_tip", result.hand_landmarks[0][16].x, result.hand_landmarks[0][16].y)
            # print("pinky_finger_tip", result.hand_landmarks[0][20].x, result.hand_landmarks[0][20].y)
            self.check_draw_position(result.hand_landmarks)
            print(self.position, self.draw)
            if self.draw and self.position == 'Active':
                self.start_draw(result)
            if self.check_change_color(result.hand_landmarks):
                self.color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
        self.result_now = result

    # displaying the image

    def start(self):
        with self.HandLandmarker.create_from_options(self.options) as landmarker:
            vid = cv2.VideoCapture("/dev/video0")
            ret, frame = vid.read()
            cv2.imshow("image", self.image_draw)
            cv2.moveWindow('image', 1000, 100)
            while True:
                ret, frame = vid.read()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.imwrite("result/result.png", image_now[:, ::-1])
                    break
                frame_timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image, timestamp_ms=frame_timestamp_ms)
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), self.result_now)
                image_now = draw_landmarks_on_image(self.image_draw, self.result_now)
                cv2.imshow('frame', annotated_image[:, ::-1])
                cv2.imshow("image", image_now[:, ::-1])

                # After the loop release the cap object
            vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
