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
from math import atan2,degrees

#import landmark_points
from landmark_points import LANDMARK_points




class LANDMARK_MATCHING(LANDMARK_points):
  def __init__(self):
    self.mp_face_mesh = mp.solutions.mediapipe.python.solutions.face_mesh
    self._landmarks = LANDMARK_points()
    
    self.size = (600, 500)
    
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

  def get_landmark_points(self, lms, ih, iw):
    points = []
    for p in lms:
      x, y = int(p.x * iw), int(p.y * ih)
      pp  = [x, y] 
      points.append(pp)
    return np.array(points)

  def landmark_pointSet_matching(self, asset_, input_):    
    res = []
    for i in range(len(asset_)):
      res.append(self.euclidean_dist_normalization(asset_[i], input_))
    index = np.argsort(res)
    return index[0], res
  
  def get_dist_idx(self, dist_list):   
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

  def AngleBtw2Points(self, pointA, pointB):
    changeInX = pointB[0] - pointA[0]
    changeInY = pointB[1] - pointA[1]
    return degrees(atan2(changeInY,changeInX))

    
  def getCenter(self, Point):
    x = Point.max_point.x - Point.min_point.x
    y = Point.max_point.y - Point.min_point.y
  
    return x, y

  def positive_cap(self, num):
    """ Cap a number to ensure positivity
    :param num: positive or negative number
    :returns: (overflow, capped_number)
    """
    if num < 0:
      return 0, abs(num)
    else:
      return num, 0

  def roi_coordinates(self, rect, size, scale):
    """ Align the rectangle into the center and return the top-left coordinates
    within the new size. If rect is smaller, we add borders.
    :param rect: (x, y, w, h) bounding rectangle of the face
    :param size: (width, height) are the desired dimensions
    :param scale: scaling factor of the rectangle to be resized
    :returns: 4 numbers. Top-left coordinates of the aligned ROI.
      (x, y, border_x, border_y). All values are > 0.
    """
    rectx, recty, rectw, recth = rect
    new_height, new_width = size
    mid_x = int((rectx + rectw/2) * scale)
    mid_y = int((recty + recth/2) * scale)
    roi_x = mid_x - int(new_width/2)
    roi_y = mid_y - int(new_height/2)

    roi_x, border_x = self.positive_cap(roi_x)
    roi_y, border_y = self.positive_cap(roi_y)
    return roi_x, roi_y, border_x, border_y

  def scaling_factor(self, rect, size):
    """ Calculate the scaling factor for the current image to be
        resized to the new dimensions
    :param rect: (x, y, w, h) bounding rectangle of the face
    :param size: (width, height) are the desired dimensions
    :returns: floating point scaling factor
    """
    new_height, new_width = size
    rect_h, rect_w = rect[2:]
    height_ratio = rect_h / new_height
    width_ratio = rect_w / new_width
    scale = 1
    if height_ratio > width_ratio:
      new_recth = 0.8 * new_height
      scale = new_recth / rect_h
    else:
      new_rectw = 0.8 * new_width
      scale = new_rectw / rect_w
    return scale

  def resize_image(self, img, scale):
    """ Resize image with the provided scaling factor
    :param img: image to be resized
    :param scale: scaling factor for resizing the image
    """
    cur_height, cur_width = img.shape[:2]
    new_scaled_height = int(scale * cur_height)
    new_scaled_width = int(scale * cur_width)

    return cv2.resize(img, (new_scaled_width, new_scaled_height))

  def resize_align(self, img, points, size):
    """ Resize image and associated points, align face to the center
      and crop to the desired size
    :param img: image to be resized
    :param points: *m* x 2 array of points
    :param size: (height, width) tuple of new desired size
    """
    new_height, new_width = size

    # Resize image based on bounding rectangle
    rect = cv2.boundingRect(np.array([points], np.int32))
    scale = self.scaling_factor(rect, size)
    img = self.resize_image(img, scale)

    # Align bounding rect to center
    cur_height, cur_width = img.shape[:2]
    roi_x, roi_y, border_x, border_y = self.roi_coordinates(rect, size, scale)
    roi_h = np.min([new_height-border_y, cur_height-roi_y])
    roi_w = np.min([new_width-border_x, cur_width-roi_x])

    # Crop to supplied size
    crop = np.zeros((new_height, new_width, 3), img.dtype)
    crop[border_y:border_y+roi_h, border_x:border_x+roi_w] = (
      img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w])

    # Scale and align face points to the crop
    points[:, 0] = (points[:, 0] * scale) + (border_x - roi_x)
    points[:, 1] = (points[:, 1] * scale) + (border_y - roi_y)

    return (crop, points)


  def get_transform(self, face_bbox, input_, asset_transform, mode):
    """
    if mode == 'LEFT_EYE': input_pts = np.array([input_[2], input_[6], input_[14], input_[29]]) 
    elif mode == 'RIGHT_EYE': input_pts = np.array([input_[2], input_[6], input_[14], input_[20]]) 
    elif mode == 'LEFT_EYE_B': input_pts = np.array([input_[2], input_[5], input_[6], input_[8]]) 
    elif mode == 'RIGHT_EYE_B': input_pts = np.array([input_[1], input_[4], input_[6], input_[8]])
    elif mode == 'NOSE': input_pts = np.array([input_[6], input_[24], input_[26], input_[56]]) 
    elif mode == 'MOUTH': input_pts = np.array([input_[0], input_[7], input_[21], input_[50]])
    """
    if mode == 'LEFT_EYE': input_pts = np.array([input_[2], input_[6], input_[14], input_[29]]) 
    elif mode == 'RIGHT_EYE': input_pts = np.array([input_[2], input_[6], input_[14], input_[20]]) 
    elif mode == 'LEFT_EYE_B': input_pts = np.array([input_[3], input_[6], input_[7], input_[8]]) 
    elif mode == 'RIGHT_EYE_B': input_pts = np.array([input_[3], input_[6], input_[7], input_[8]])
    elif mode == 'NOSE': input_pts = np.array([input_[6], input_[24], input_[26], input_[56]]) 
    elif mode == 'MOUTH': input_pts = np.array([input_[0], input_[7], input_[21], input_[50]])

    
    input_bbox = BoundingBox(input_)
    input_angle = self.getAngle_Dist_2P(input_pts[0], input_pts[2])

    input_w_dist = (input_bbox.max_point.x - input_bbox.min_point.x) / (face_bbox.max_point.x - face_bbox.min_point.x) 
    input_h_dist = (input_bbox.max_point.y - input_bbox.min_point.y) / (face_bbox.max_point.y - face_bbox.min_point.y) 

    input_center_x, input_center_y = self.getCenter(input_bbox)


    return (input_angle-asset_transform[0]), (input_w_dist/asset_transform[1]), (input_h_dist/asset_transform[2]), (input_center_x-asset_transform[3]), (input_center_y-asset_transform[4])
  


  def value_to_list(self, lists, Angle, w_scale, h_scale, w_trans, h_trans):
    lists.append(Angle)
    lists.append(w_scale)
    lists.append(h_scale)
    lists.append(w_trans)
    lists.append(h_trans)  


  def _RotatePoint(self, p, rad):
    x = math.cos(rad) * p[0] - math.sin(rad) * p[1]
    y = math.sin(rad) * p[0] + math.cos(rad) * p[1]
    return [x, y]

  def RotatePoint(self, cen_pt, p, rad):
    trans_pt = p - cen_pt
    rot_pt = self._RotatePoint(trans_pt, rad)
    fin_pt = rot_pt + cen_pt
    return fin_pt


  def GetRadian(self, input_image, p1, p2):
    ih, iw, _ = input_image.shape 

    self.anchorX = abs(p1[0] - iw/2)
    self.anchorY = abs(p1[1] - ih/2)

    M = np.float32([[1, 0, - self.anchorX], [0, 1, - self.anchorY]]) 
    #img_translation = cv2.warpAffine(input_image, M, (iw, ih))

    points=[]
    points.append((p1[0] - self.anchorX, p1[1] - self.anchorY))
    points.append((p2[0] - self.anchorX, p2[1] - self.anchorY))

    angle = self.AngleBtw2Points(points[0], points[1]) + 90
    M = cv2.getRotationMatrix2D((points[0][0], points[0][1]), angle, 1)
    #img_rotation = cv2.warpAffine(img_translation, M, (iw, ih))

    rad = angle * (math.pi / 180.0)

    return rad, angle#, img_rotation



  def TotalResult(self, assets, inputs, transforms):
      Face_contour_ID = 0 #, _ = self.landmark_pointSet_matching(assets[0], inputs[0])
      _, res_l_eye = self.landmark_pointSet_matching(assets[1], inputs[1])
      _, res_r_eye = self.landmark_pointSet_matching(assets[2], inputs[2])      
      Eye_ID = self.dist_sum(res_l_eye, res_r_eye)
      _, res_l_eye = self.landmark_pointSet_matching(assets[3], inputs[3])
      _, res_r_eye = self.landmark_pointSet_matching(assets[4], inputs[4])
      Eye_B_ID = self.dist_sum(res_l_eye, res_r_eye)
      Nose_ID, _ = self.landmark_pointSet_matching(assets[5], inputs[5])
      Mouth_ID, _ = self.landmark_pointSet_matching(assets[6], inputs[6])


      Face_contour=[]
      Nose=[]
      L_Eye = []
      R_Eye = []
      L_Eye_b = []
      R_Eye_b = []
      Mouth = []

      face_bbox = BoundingBox(inputs[0])
      self.value_to_list(Face_contour, 0.0, 1.0, 1.0, 0.0, 0.0)

      #Nose_ID, Eye_ID, Eye_B_ID, Mouth_ID = 1, 1, 1, 1


      Angle, h_scale, v_scale, h_trans, v_trans = self.get_transform(face_bbox, inputs[5], transforms[0][Nose_ID], 'NOSE')
      self.value_to_list(Nose, 0, h_scale, 1, 0, 0)
      
      Angle_l, h_scale_l, v_scale_l, h_trans_l, v_trans_l = self.get_transform(face_bbox, inputs[1], transforms[1][Eye_ID], 'LEFT_EYE')
      Angle_r, h_scale_r, v_scale_r, h_trans_r, v_trans_r = self.get_transform(face_bbox, inputs[2], transforms[2][Eye_ID], 'RIGHT_EYE')
      
      if (h_trans_l + h_trans_r)/2 > 10:
        h_trans_l, h_trans_r = 10, 10
      elif (h_trans_l + h_trans_r)/2 < 10:
        h_trans_l, h_trans_r = -10, -10

      if (v_trans_l + v_trans_r)/2 > 10:
        v_trans_l, v_trans_r = 10, 10
      elif (v_trans_l + v_trans_r)/2 < 10:
        v_trans_l, v_trans_r = -10, -10

      if Angle_l < Angle_r:
        self.value_to_list(L_Eye, Angle_l, (h_scale_l + h_scale_r) /2, (v_scale_l + v_scale_r)/2, (h_trans_l + h_trans_r)/2, (v_trans_l + v_trans_r) /2)
        self.value_to_list(R_Eye, -Angle_l, (h_scale_l + h_scale_r) /2, (v_scale_l + v_scale_r)/2, -(h_trans_l + h_trans_r)/2, (v_trans_l + v_trans_r) /2)
      else:
        self.value_to_list(L_Eye, -Angle_r, (h_scale_l + h_scale_r) /2, (v_scale_l + v_scale_r)/2, -(h_trans_l + h_trans_r)/2, (v_trans_l + v_trans_r) /2)
        self.value_to_list(R_Eye, Angle_r, (h_scale_l + h_scale_r) /2, (v_scale_l + v_scale_r)/2, (h_trans_l + h_trans_r)/2, (v_trans_l + v_trans_r) /2)

      Angle_l, h_scale_l, v_scale_l, h_trans_l, v_trans_l = self.get_transform(face_bbox, inputs[3], transforms[3][Eye_B_ID], 'LEFT_EYE_B')
      Angle_r, h_scale_r, v_scale_r, h_trans_r, v_trans_r = self.get_transform(face_bbox, inputs[4], transforms[4][Eye_B_ID], 'RIGHT_EYE_B')

      if (h_trans_l + h_trans_r)/2 > 10:
        h_trans_l, h_trans_r = 10, 10
      elif (h_trans_l + h_trans_r)/2 < 10:
        h_trans_l, h_trans_r = -10, -10

      if (v_trans_l + v_trans_r)/2 > 10:
        v_trans_l, v_trans_r = 10, 10
      elif (v_trans_l + v_trans_r)/2 < 10:
        v_trans_l, v_trans_r = -10, -10

      if Angle_l < Angle_r:
        self.value_to_list(L_Eye_b, Angle_l, (h_scale_l + h_scale_r) /2, (v_scale_l + v_scale_r)/2, (h_trans_l + h_trans_r)/2, (v_trans_l + v_trans_r) /2)
        self.value_to_list(R_Eye_b, -Angle_l, (h_scale_l + h_scale_r) /2, (v_scale_l + v_scale_r)/2, -(h_trans_l + h_trans_r)/2, (v_trans_l + v_trans_r) /2)
      else:
        self.value_to_list(L_Eye_b, -Angle_r, (h_scale_l + h_scale_r) /2, (v_scale_l + v_scale_r)/2, -(h_trans_l + h_trans_r)/2, (v_trans_l + v_trans_r) /2)
        self.value_to_list(R_Eye_b, Angle_r, (h_scale_l + h_scale_r) /2, (v_scale_l + v_scale_r)/2, (h_trans_l + h_trans_r)/2, (v_trans_l + v_trans_r) /2)


      Angle, h_scale, v_scale, h_trans, v_trans  = self.get_transform(face_bbox, inputs[6], transforms[5][Mouth_ID], 'MOUTH')
      self.value_to_list(Mouth, 0, h_scale, v_scale, 0, v_trans)
      
      transform_ = (Face_contour, L_Eye_b, R_Eye_b, Nose, Mouth, L_Eye, R_Eye)
      
      return [Face_contour_ID, Eye_B_ID, Eye_B_ID, Nose_ID, Mouth_ID, Eye_ID, Eye_ID], transform_



  #mode:0 woman, 1 man
  def landmark_part_matching(self, mode, input_image):
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

    ih, iw, ic = input_image.shape

    if results.multi_face_landmarks:
        o_points = self.get_landmark_points(results.multi_face_landmarks[0].landmark, ih, iw)
        input_image, points = self.resize_align(input_image, o_points, self.size)
        
        ih, iw, ic = input_image.shape
        rad, angle= self.GetRadian(input_image, points[4], points[8])
        
        for id, lm in enumerate(points):
          #if abs(angle) > 5.0:
          #  p = self.RotatePoint(np.array((int(iw/2), int(ih/2))), [lm[0] - self.anchorX , lm[1] - self.anchorY], -rad)
          #  (x, y) = (int(p[0]), int(p[1]))
          #else:
          (x, y) = (lm[0], lm[1])
            
          if id in self._landmarks.FACE_CONTOUR : 
            input_Face_contour.append((x,y))
          elif id in self._landmarks.LEFT_EYE : 
            input_left_eye.append((x, y))
          elif id in self._landmarks.RIGHT_EYE : 
            input_right_eye.append((x, y))
          elif id in self._landmarks.LEFT_EYE_B : 
            input_left_eye_b.append((x, y)) 
          elif id in self._landmarks.RIGHT_EYE_B : 
            input_right_eye_b.append((x, y))
          elif id in self._landmarks.NOSE : 
            input_nose.append((x, y))
          elif id in self._landmarks.MOUTH : 
            input_mouth.append((x, y))
          
        
        inputs = [input_Face_contour, input_left_eye, input_right_eye, input_left_eye_b, input_right_eye_b, input_nose, input_mouth]
        
        # WOAMN
        if mode == 0:
          assets = [self._landmarks.Asset_w_Face_contours, self._landmarks.Asset_w_left_eyes, self._landmarks.Asset_w_right_eyes, self._landmarks.Asset_w_left_eyes_b, self._landmarks.Asset_w_right_eyes_b, self._landmarks.Asset_w_nose, self._landmarks.Asset_w_mouths]
          transforms = [self._landmarks.Asset_w_nose_transform, self._landmarks.Asset_w_left_eyes_transform, self._landmarks.Asset_w_right_eyes_transform, self._landmarks.Asset_w_left_eyes_b_transform, self._landmarks.Asset_w_right_eyes_b_transform, self._landmarks.Asset_w_mouths_transform]
        # MAN
        else:
          assets = [self._landmarks.Asset_m_Face_contours, self._landmarks.Asset_m_left_eyes, self._landmarks.Asset_m_right_eyes, self._landmarks.Asset_m_left_eyes_b, self._landmarks.Asset_m_right_eyes_b, self._landmarks.Asset_m_nose, self._landmarks.Asset_m_mouths]
          transforms = [self._landmarks.Asset_m_nose_transform, self._landmarks.Asset_m_left_eyes_transform, self._landmarks.Asset_m_right_eyes_transform, self._landmarks.Asset_m_left_eyes_b_transform, self._landmarks.Asset_m_right_eyes_b_transform, self._landmarks.Asset_m_mouths_transform]

        return self.TotalResult(assets, inputs, transforms)
    else:
        return [], []










  def get_transform_DB(self, face_bbox, input_, mode):
    if mode == 'LEFT_EYE': input_pts = np.array([input_[2], input_[6], input_[14], input_[29]]) 
    elif mode == 'RIGHT_EYE': input_pts = np.array([input_[2], input_[6], input_[14], input_[20]]) 
    elif mode == 'LEFT_EYE_B': input_pts = np.array([input_[2], input_[5], input_[6], input_[8]]) 
    elif mode == 'RIGHT_EYE_B': input_pts = np.array([input_[1], input_[4], input_[6], input_[8]])
    elif mode == 'NOSE': input_pts = np.array([input_[6], input_[24], input_[26], input_[56]]) 
    elif mode == 'MOUTH': input_pts = np.array([input_[0], input_[7], input_[21], input_[50]])

    input_bbox = BoundingBox(input_)
    input_angle = self.getAngle_Dist_2P(input_pts[0], input_pts[2])

    input_w_dist = (input_bbox.max_point.x - input_bbox.min_point.x) / (face_bbox.max_point.x - face_bbox.min_point.x) 
    input_h_dist = (input_bbox.max_point.y - input_bbox.min_point.y) / (face_bbox.max_point.y - face_bbox.min_point.y) 

    input_center_x, input_center_y = self.getCenter(input_bbox)
    
    return input_angle, input_w_dist, input_h_dist, input_center_x, input_center_y


  def makeDatabase(self, input_image, id_data):
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
        o_points = self.get_landmark_points(results.multi_face_landmarks[0].landmark, ih, iw)
        input_image, points = self.resize_align(input_image, o_points, self.size)
        
        ih, iw, ic = input_image.shape
        #rad, angle = self.GetRadian(input_image, points[4], points[8])

        for id, lm in enumerate(points):
          #if abs(angle) > 5.0:
          #  p = self.RotatePoint(np.array((int(iw/2), int(ih/2))), [lm[0] - self.anchorX , lm[1] - self.anchorY], -rad)
          #  (x, y) = int(p[0]), int(p[1])
          #else:
          (x, y) = (lm[0], lm[1])
            
          if id in self._landmarks.FACE_CONTOUR : 
            input_Face_contour.append((x,y))
          elif id in self._landmarks.LEFT_EYE : 
            input_left_eye.append((x, y))
          elif id in self._landmarks.RIGHT_EYE : 
            input_right_eye.append((x, y))
          elif id in self._landmarks.LEFT_EYE_B : 
            input_left_eye_b.append((x, y))
          elif id in self._landmarks.RIGHT_EYE_B : 
            input_right_eye_b.append((x, y))
          elif id in self._landmarks.NOSE :
            input_nose.append((x, y))
          elif id in self._landmarks.MOUTH : 
            input_mouth.append((x, y))

    
        print(f'Asset_{id_data}_FACE_CONTOUR =',input_Face_contour)
        print(f'Asset_{id_data}_LEFT_EYE =',input_left_eye)
        print(f'Asset_{id_data}_RIGHT_EYE =',input_right_eye)
        print(f'Asset_{id_data}_LEFT_EYE_B =',input_left_eye_b)
        print(f'Asset_{id_data}_RIGHT_EYE_B =',input_right_eye_b)
        print(f'Asset_{id_data}_NOSE =',input_nose)
        print(f'Asset_{id_data}_MOUTH =',input_mouth)
        print()

        transforms = []

        face_bbox = BoundingBox(input_Face_contour)

        Angle, h_scale, v_scale, h_trans, v_trans  = self.get_transform_DB(face_bbox, input_nose, 'NOSE')
        transforms.append([Angle, h_scale, v_scale, h_trans, v_trans])
        Angle, h_scale, v_scale, h_trans, v_trans  = self.get_transform_DB(face_bbox, input_left_eye, 'LEFT_EYE')
        transforms.append([Angle, h_scale, v_scale, h_trans, v_trans])
        Angle, h_scale, v_scale, h_trans, v_trans  = self.get_transform_DB(face_bbox, input_right_eye, 'RIGHT_EYE')
        transforms.append([Angle, h_scale, v_scale, h_trans, v_trans])
        Angle, h_scale, v_scale, h_trans, v_trans  = self.get_transform_DB(face_bbox, input_left_eye_b, 'LEFT_EYE_B')
        transforms.append([Angle, h_scale, v_scale, h_trans, v_trans])
        Angle, h_scale, v_scale, h_trans, v_trans  = self.get_transform_DB(face_bbox, input_right_eye_b, 'RIGHT_EYE_B')
        transforms.append([Angle, h_scale, v_scale, h_trans, v_trans])
        Angle, h_scale, v_scale, h_trans, v_trans  = self.get_transform_DB(face_bbox, input_mouth, 'MOUTH')
        transforms.append([Angle, h_scale, v_scale, h_trans, v_trans])

        print(f'Asset_{id_data}_TRANSFORM =', transforms)




if __name__ == "__main__":
  mode = 1

  id = 0
  input_image = cv2.imread(f"{id}.png")
  #input_image = cv2.imread("1.png")

  land = LANDMARK_MATCHING()
  input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  
  #land.makeDatabase(input_image, id)
  res, transform  = land.landmark_part_matching(mode, input_image)

  # Make json result
  # TODO add rotation, scale, translation value
  data = []
  for i, num in enumerate(res):
      if mode == 0:
          if i == 0: asset_name = land._landmarks.WOMAN_FACE_NAME[num]
          elif i == 1 or i == 2: asset_name = land._landmarks.WOMAN_EYE_BROW_NAME[num]
          elif i == 3: asset_name = land._landmarks.WOMAN_NOSE_NAME[num]
          elif i == 4: asset_name = land._landmarks.WOMAN_MOUTH_NAME[num]
          else: asset_name = land._landmarks.WOMAN_EYE_NAME[num]
      else:
          if i == 0: asset_name = land._landmarks.MAN_FACE_NAME[num]
          elif i == 1 or i == 2: asset_name = land._landmarks.MAN_EYE_BROW_NAME[num]
          elif i == 3: asset_name = land._landmarks.MAN_NOSE_NAME[num]
          elif i == 4: asset_name = land._landmarks.MAN_MOUTH_NAME[num]
          else: asset_name = land._landmarks.MAN_EYE_NAME[num]

      data.append({"type":int(i), "asset_name":asset_name, "rotation":transform[i][0], "h_scale":transform[i][1], "v_scale":transform[i][2], "h_trans":transform[i][3], "v_trans":transform[i][4]})

  print ({'status': 'success' , 'data': data})