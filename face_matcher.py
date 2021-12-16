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

  def get_landmark_point(self, lm, ih, iw ):
    x, y = int(lm.x * iw), int(lm.y * ih)
    return (x, y)

  def landmark_pointSet_matching(self, asset_, input_):    
    res = []
    for i in range(self._landmarks.Asset_size):
      res.append(self.euclidean_dist_normalization(asset_[i], input_))
    index = np.argsort(res)
    return index[0], res
  
  def get_dist_idx(self, dist_list):   
    print(dist_list) 
    index = np.argsort(dist_list)
    return index[0]

  def euclidean_dist_normalization(self, asset_, input_):
    input_ = np.reshape(np.array(input_), (len(input_)*2, 1))   
    input_ = self.normalization(input_)
    
    asset_ = np.reshape(np.array(asset_), (len(asset_)*2, 1))
    asset_ = self.normalization(asset_)
    res = distance.euclidean(asset_, input_)
    return res
    
  def euclidean_dist(self, asset_, input_):
    res = distance.euclidean(asset_, input_)
    return res

  def dist_sum(self, a, b):
    res = [a[i]+b[i] for i in range(len(a))]
    index = np.argsort(res)
    return index[0]

  # Two points angle: clockWise (CW)
  def getAngle_Dist_2P(self, p1, p2, direction="CW"):  
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    if direction == "CCW": res = (360 - res) % 360
    return res
    
  def getCenter(self, Point):
    x = Point.max_point.x - Point.min_point.x
    y = Point.max_point.y - Point.min_point.y
  
    return x, y

  def get_transform(self, input_, asset_, idx, mode):

    if mode == 'LEFT_EYE': input_pts = np.array([input_[3], input_[1], input_[2], input_[0]]) 
    elif mode == 'RIGHT_EYE': input_pts = np.array([input_[2], input_[1], input_[3], input_[0]]) 
    elif mode == 'LEFT_EYE_B': input_pts = np.array([input_[2], input_[3], input_[0], input_[1]]) 
    elif mode == 'RIGHT_EYE_B': input_pts = np.array([input_[3], input_[2], input_[1], input_[0]])
    elif mode == 'NOSE': input_pts = np.array([input_[1], input_[0], input_[3], input_[2]]) 
    elif mode == 'MOUTH': input_pts = np.array([input_[2], input_[0], input_[3], input_[1]]) 

    input_bbox = BoundingBox(input_pts)
    input_angle = self.getAngle_Dist_2P(input_pts[0], input_pts[2])

    asset_pts = np.array(asset_[idx])  
    asset_angle = self.getAngle_Dist_2P(asset_pts[0], asset_pts[2])
    asset_bbox = BoundingBox(asset_pts)


    if mode == 'LEFT_EYE' or mode == 'RIGHT_EYE' or mode == 'NOSE' or mode == 'MOUTH' :
      input_w_dist = self.euclidean_dist(input_pts[0], input_pts[2])
      input_h_dist = self.euclidean_dist(input_pts[1], input_pts[3])
      asset_w_dist = self.euclidean_dist(asset_pts[0], asset_pts[2])
      asset_h_dist = self.euclidean_dist(asset_pts[1], asset_pts[3])
    elif mode == 'LEFT_EYE_B' or mode == 'RIGHT_EYE_B':
      input_w_dist = self.euclidean_dist(input_pts[0], input_pts[1])
      input_h_dist = self.euclidean_dist(input_pts[0], input_pts[3])
      asset_w_dist = self.euclidean_dist(asset_pts[0], asset_pts[1])
      asset_h_dist = self.euclidean_dist(asset_pts[0], asset_pts[3])
    
    input_center_x, input_center_y = self.getCenter(input_bbox)
    asset_center_x, asset_center_y = self.getCenter(asset_bbox)

    return input_angle-asset_angle, input_w_dist/asset_w_dist, input_h_dist/asset_h_dist, asset_center_x-input_center_x, asset_center_y-input_center_y
  
  def value_to_list(self, lists, Angle, w_scale, h_scale, w_trans, h_trans):
    lists.append(Angle)
    lists.append(w_scale)
    lists.append(h_scale)
    lists.append(w_trans)
    lists.append(h_trans)  

  def landmark_part_matching(self, input_image):
    results = self.face_mesh.process(input_image)

    # Draw the face mesh annotations on the image.
    input_image.flags.writeable = True
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    input_Face_contour = []
    input_left_eye = []
    input_right_eye = []
    input_left_eye_b = []
    input_right_eye_b = []
    input_nose = []
    input_mouth = []
     
    
    transform_input_left_eye = []
    transform_input_right_eye = []
    transform_input_left_eye_b = []
    transform_input_right_eye_b = []
    transform_input_nose = []
    transform_input_mouth = []

    ih, iw, ic = input_image.shape
    if results.multi_face_landmarks:
      for faceLms in results.multi_face_landmarks:
          for id, lm in enumerate(faceLms.landmark):            
            if id in self._landmarks.FACE_CONTOUR : input_Face_contour.append(self.get_landmark_point(lm, ih, iw))
            if id in self._landmarks.LEFT_EYE : 
              (x, y) = self.get_landmark_point(lm, ih, iw)
              input_left_eye.append((x, y))                
              if id in self._landmarks.TRANSFORM_LEFT_EYE : transform_input_left_eye.append((x, y))
              
            if id in self._landmarks.RIGHT_EYE : 
              (x, y) = self.get_landmark_point(lm, ih, iw)
              input_right_eye.append((x, y))
              if id in self._landmarks.TRANSFORM_RIGHT_EYE : transform_input_right_eye.append((x, y))

            if id in self._landmarks.LEFT_EYE_B : 
              (x, y) = self.get_landmark_point(lm, ih, iw)
              input_left_eye_b.append((x, y))
              if id in self._landmarks.TRANSFORM_LEFT_EYE_B : transform_input_left_eye_b.append((x, y))
              
            if id in self._landmarks.RIGHT_EYE_B : 
              (x, y) = self.get_landmark_point(lm, ih, iw)
              input_right_eye_b.append((x, y))
              if id in self._landmarks.TRANSFORM_RIGHT_EYE_B : transform_input_right_eye_b.append((x, y))

            if id in self._landmarks.NOSE : 
              (x, y) = self.get_landmark_point(lm, ih, iw)
              input_nose.append((x, y))
              if id in self._landmarks.TRANSFORM_NOSE : transform_input_nose.append((x, y))

            if id in self._landmarks.MOUTH : 
              (x, y) = self.get_landmark_point(lm, ih, iw)
              input_mouth.append((x, y))
              if id in self._landmarks.TRANSFORM_MOUTH : transform_input_mouth.append((x, y))

    Face_contour_ID, _ = self.landmark_pointSet_matching(self._landmarks.Asset_Face_contours, input_Face_contour)
    _, res_l_eye = self.landmark_pointSet_matching(self._landmarks.Asset_left_eyes, input_left_eye)
    _, res_r_eye = self.landmark_pointSet_matching(self._landmarks.Asset_right_eyes, input_right_eye)      
    Eye_ID = self.dist_sum(res_l_eye, res_r_eye)
    _, res_l_eye = self.landmark_pointSet_matching(self._landmarks.Asset_left_eyes_b, input_left_eye_b)
    _, res_r_eye = self.landmark_pointSet_matching(self._landmarks.Asset_right_eyes_b, input_right_eye_b)
    Eye_B_ID = self.dist_sum(res_l_eye, res_r_eye)
    Nose_ID, _ = self.landmark_pointSet_matching(self._landmarks.Asset_nose, input_nose)
    Mouth_ID, _ = self.landmark_pointSet_matching(self._landmarks.Asset_mouths, input_mouth)

    Face_contour=[]
    Nose=[]
    L_Eye = []
    R_Eye = []
    L_Eye_b = []
    R_Eye_b = []
    Mouth = []
    
    self.value_to_list(Face_contour, 0.0, 1.0, 1.0, 0.0, 0.0)

    Angle, v_scale, h_scale, v_trans, h_trans  = self.get_transform(transform_input_nose, self._landmarks.Asset_transform_nose, Nose_ID, 'NOSE')
    self.value_to_list(Nose, Angle, h_scale, v_scale, h_trans, v_trans)
    
    Angle, v_scale, h_scale, v_trans, h_trans  = self.get_transform(transform_input_left_eye, self._landmarks.Asset_transform_left_eyes, Eye_ID, 'LEFT_EYE')
    self.value_to_list(L_Eye, Angle, h_scale, v_scale, h_trans, v_trans)

    Angle, v_scale, h_scale, v_trans, h_trans  = self.get_transform(transform_input_right_eye, self._landmarks.Asset_transform_right_eyes, Eye_ID, 'RIGHT_EYE')
    self.value_to_list(R_Eye, Angle, h_scale, v_scale, h_trans, v_trans)

    Angle, v_scale, h_scale, v_trans, h_trans  = self.get_transform(transform_input_left_eye_b, self._landmarks.Asset_transform_left_eyes_b, Eye_B_ID, 'LEFT_EYE_B')
    self.value_to_list(L_Eye_b, Angle, h_scale, v_scale, h_trans, v_trans)

    Angle, v_scale, h_scale, v_trans, h_trans  = self.get_transform(transform_input_right_eye_b, self._landmarks.Asset_transform_right_eyes_b, Eye_B_ID, 'RIGHT_EYE_B')
    self.value_to_list(R_Eye_b, Angle, h_scale, v_scale, h_trans, v_trans)

    Angle, v_scale, h_scale, v_trans, h_trans  = self.get_transform(transform_input_mouth, self._landmarks.Asset_transform_mouths, Mouth_ID, 'MOUTH')
    self.value_to_list(Mouth, Angle, h_scale, v_scale, h_trans, v_trans)
    
    transform_ = (Face_contour, Nose, L_Eye, R_Eye, L_Eye_b, R_Eye_b, Mouth)
    return [Face_contour_ID, Nose_ID, Eye_ID,  Eye_ID, Eye_B_ID, Eye_B_ID, Mouth_ID], transform_

