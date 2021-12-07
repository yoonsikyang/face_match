import cv2
import mediapipe as mp

import math
from typing import List, Mapping, Optional, Tuple, Union

import cv2
import dataclasses
import matplotlib.pyplot as plt
import numpy as np


from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2
from mediapipe.framework.formats import landmark_pb2

from scipy.spatial import distance

from planar import BoundingBox

#import landmark_points
from landmark_points import LANDMARK_points


class LANDMARK_MATCHING (LANDMARK_points):
  def __init__(self):
    self.mp_face_mesh = mp.solutions.mediapipe.python.solutions.face_mesh
    self._landmarks = LANDMARK_points()
    self.face_mesh =  self.mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

  
  def normalization(self, x):
      min_value = min(x)
      max_value = max(x) 
      return list(map(lambda x: (x-min_value)/(max_value-min_value), x))

  def get_landmark_point(self, mode_, lm, ih, iw, ic ):
    x, y = int(lm.x * iw), int(lm.y * ih)
    return (x, y)

  def get_matching_idx(self, asset_, input_):    
    res = []
    
    input_point = np.reshape(np.array(input_), (len(input_)*2, 1))
    input_point = self.normalization(input_point)

    for i in range(self._landmarks.Asset_size):
      asset_point = np.reshape(np.array(asset_[i]), (len(asset_[i])*2, 1))
      asset_point = self.normalization(asset_point)
    
      res.append(distance.euclidean(asset_point, input_point))
    print(res)
    index = np.argsort(res)
    #index = index[::-1] #from large to small
    return index[0], res

  def eye_res_sum(self, a, b):
    res = [a[i]+b[i] for i in range(len(a))]
    index = np.argsort(res)
    return index[0], index[0]


  def landmark_part_matching(self, image):
    results = self.face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    input_Face_contour = []
    input_left_eye = []
    input_right_eye = []
    input_left_eye_b = []
    input_right_eye_b = []
    input_nose = []
    input_mouth = []

    ih, iw, ic = image.shape
    if results.multi_face_landmarks:
      for faceLms in results.multi_face_landmarks:
          for id, lm in enumerate(faceLms.landmark):
            if id in self._landmarks.FACE_CONTOUR : input_Face_contour.append(self.get_landmark_point(self._landmarks.FACE_CONTOUR, lm, ih, iw, ic))
            if id in self._landmarks.LEFT_EYE : input_left_eye.append(self.get_landmark_point(self._landmarks.LEFT_EYE, lm, ih, iw, ic))
            if id in self._landmarks.RIGHT_EYE : input_right_eye.append(self.get_landmark_point(self._landmarks.RIGHT_EYE, lm, ih, iw, ic))
            if id in self._landmarks.LEFT_EYE_B : input_left_eye_b.append(self.get_landmark_point(self._landmarks.LEFT_EYE_B, lm, ih, iw, ic))
            if id in self._landmarks.RIGHT_EYE_B : input_right_eye_b.append(self.get_landmark_point(self._landmarks.RIGHT_EYE_B, lm, ih, iw, ic))
            if id in self._landmarks.NOSE : input_nose.append(self.get_landmark_point(self._landmarks.NOSE, lm, ih, iw, ic))
            if id in self._landmarks.MOUTH : input_mouth.append(self.get_landmark_point(self._landmarks.MOUTH, lm, ih, iw, ic))
          
    
      res_face_contour_id, _ = self.get_matching_idx(self._landmarks.Face_contours, input_Face_contour)
      _, res_l_eye = self.get_matching_idx(self._landmarks.left_eyes, input_left_eye)
      _, res_r_eye = self.get_matching_idx(self._landmarks.right_eyes, input_right_eye)      
      res_left_eye_id, res_right_eye_id = self.eye_res_sum(res_l_eye, res_r_eye)
      _, res_l_eye = self.get_matching_idx(self._landmarks.left_eyes_b, input_left_eye_b)
      _, res_r_eye = self.get_matching_idx(self._landmarks.right_eyes_b, input_right_eye_b)
      res_left_eye_b_id, res_right_eye_b_id = self.eye_res_sum(res_l_eye, res_r_eye)
      res_nose_id, _ = self.get_matching_idx(self._landmarks.nose, input_nose)
      res_mouth_id, _ = self.get_matching_idx(self._landmarks.mouths, input_mouth)
      print("-------------")
      print("Face_contour_id", res_face_contour_id)
      print("Left_eye_id", res_left_eye_id)
      print("Right_eye_id", res_right_eye_id)
      print("Left_eye_b_id", res_left_eye_b_id)
      print("Right_eye_b_id", res_right_eye_b_id)
      print("Nose_id", res_nose_id)
      print("Mouth_id", res_mouth_id)
      return [res_face_contour_id, res_left_eye_id,  res_right_eye_id, res_left_eye_b_id, res_right_eye_b_id, res_nose_id, res_mouth_id]
    
    else:
      return []

    
"""
if __name__ == "__main__":
  image = cv2.imread("images/test/2/2-8.jpg")
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  land = LANDMARK_MATCHING()
  res = land.landmark_part_matching(image)

  print(res)
"""