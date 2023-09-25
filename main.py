# # import the opencv library
import cv2
import time
from draw_hand import draw_landmarks_on_image
import numpy as np
""" vid = cv2.VideoCapture("/dev/video3")

while(True):
        print(vid.get(cv2.CAP_PROP_POS_MSEC))
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
vid.release()
# Destroy all the windows
cv2.destroyAllWindows() """
  
# define a video capture object


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DetectionHand():
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.result_now = self.HandLandmarkerResult(handedness=[], hand_landmarks=[], hand_world_landmarks=[])
        self.options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=self.make_result)
        self.image_draw = np.full((500, 500, 3),
                            255, dtype = np.uint8)
        self.image_draw_height = 500
        self.image_draw_width = 500

    def check_valid_position(self,hand_landmarks):
        index_finger_tip_y=hand_landmarks[0][8].y
        middle_finger_tip_y= hand_landmarks[0][12].y
        ring_finger_tip_y= hand_landmarks[0][16].y
        pinky_finger_tip_y= hand_landmarks[0][20].y
        if middle_finger_tip_y-index_finger_tip_y>0.30:
            return True
        return False

# Create a hand landmarker instance with the live stream mode:
    def make_result(self,result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """ print('---------------------------------hand landmarker result: {}'.format(result)) """

        if(result.hand_landmarks!=[]):
            print("index_finger_tip",result.hand_landmarks[0][8].x,result.hand_landmarks[0][8].y)
            print("middle_finger_tip",result.hand_landmarks[0][12].x,result.hand_landmarks[0][12].y)
            print("ring_finger_tip",result.hand_landmarks[0][16].x,result.hand_landmarks[0][16].y)
            print("pinky_finger_tip",result.hand_landmarks[0][20].x,result.hand_landmarks[0][20].y)
            if (self.check_valid_position(result.hand_landmarks)):
                draw_x = result.hand_landmarks[0][8].x
                draw_y = result.hand_landmarks[0][8].y
                draw_x_normal = draw_x*self.image_draw_width
                draw_y_normal = draw_y*self.image_draw_height
                self.image_draw = cv2.circle(self.image_draw, (int(draw_x_normal), int(draw_y_normal)), radius=4, color=(0, 0, 255), thickness=-1)
            
        self.result_now = result
    # displaying the image

    def start(self):
        with self.HandLandmarker.create_from_options(self.options) as landmarker:
            vid = cv2.VideoCapture("/dev/video2")
            ret, frame = vid.read()
            cv2.imshow("image", self.image_draw)
            cv2.moveWindow('image', 1000, 100)
            while(True):
                ret, frame = vid.read()
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)  
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame_timestamp_ms = int(time.time() * 1000)
                landmarker.detect_async(mp_image,timestamp_ms=frame_timestamp_ms)
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), self.result_now)
                cv2.imshow('frame',annotated_image[:,::-1])
                cv2.imshow("image", self.image_draw[:,::-1])

                # After the loop release the cap object
            vid.release()
            # Destroy all the windows
            cv2.destroyAllWindows()

Detection_model= DetectionHand()
Detection_model.start()