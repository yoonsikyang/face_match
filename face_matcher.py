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


  def get_transform(self, input_, asset_, idx, mode):

    if mode == 'LEFT_EYE': input_pts = np.array([input_[0], input_[1], input_[2], input_[3]]) 
    elif mode == 'RIGHT_EYE': input_pts = np.array([input_[0], input_[1], input_[2], input_[3]]) 
    elif mode == 'LEFT_EYE_B': input_pts = np.array([input_[0], input_[1], input_[2], input_[3]]) 
    elif mode == 'RIGHT_EYE_B': input_pts = np.array([input_[0], input_[1], input_[2], input_[3]])
    elif mode == 'NOSE': input_pts = np.array([input_[0], input_[1], input_[2], input_[3]]) 
    elif mode == 'MOUTH': input_pts = np.array([input_[0], input_[3], input_[4], input_[5], input_[1], input_[2]]) 

    input_bbox = BoundingBox(input_pts)
    input_angle = self.getAngle_Dist_2P(input_pts[0], input_pts[2])

    asset_pts = np.array(asset_[idx])  
    asset_angle = self.getAngle_Dist_2P(asset_pts[0], asset_pts[2])
    asset_bbox = BoundingBox(asset_pts)

    if mode == 'LEFT_EYE' or mode == 'RIGHT_EYE' or mode == 'NOSE':
      input_w_dist = self.euclidean_dist(input_pts[0], input_pts[2])
      input_h_dist = self.euclidean_dist(input_pts[1], input_pts[3])
      asset_w_dist = self.euclidean_dist(asset_pts[0], asset_pts[2])
      asset_h_dist = self.euclidean_dist(asset_pts[1], asset_pts[3])
    elif mode == 'LEFT_EYE_B' or mode == 'RIGHT_EYE_B':
      input_w_dist = self.euclidean_dist(input_pts[0], input_pts[1])
      input_h_dist = self.euclidean_dist(input_pts[0], input_pts[3])
      asset_w_dist = self.euclidean_dist(asset_pts[0], asset_pts[1])
      asset_h_dist = self.euclidean_dist(asset_pts[0], asset_pts[3])
    elif mode == 'MOUTH':
      input_w_dist = self.euclidean_dist(input_pts[0], input_pts[2])
      input_h_dist = self.euclidean_dist(input_pts[1], input_pts[3]) - self.euclidean_dist(input_pts[5], input_pts[4])
      asset_w_dist = self.euclidean_dist(asset_pts[0], asset_pts[2])
      asset_h_dist = self.euclidean_dist(asset_pts[1], asset_pts[3])
      print('ssssssssssssssssssssssssssss ',self.euclidean_dist(input_pts[5], input_pts[4]))

    input_center_x, input_center_y = self.getCenter(input_bbox)
    asset_center_x, asset_center_y = self.getCenter(asset_bbox)
    #Angle, v_scale, h_scale, v_trans, h_trans
    return input_angle-asset_angle, (input_w_dist/asset_w_dist), (input_h_dist/asset_h_dist), (asset_center_x-input_center_x), (asset_center_y-input_center_y)
  
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
    img_translation = cv2.warpAffine(input_image, M, (iw, ih))

    points=[]
    points.append((p1[0] - self.anchorX, p1[1] - self.anchorY))
    points.append((p2[0] - self.anchorX, p2[1] - self.anchorY))

    angle = self.AngleBtw2Points(points[0], points[1]) + 90

    M = cv2.getRotationMatrix2D((points[0][0], points[0][1]), angle, 1)
    img_rotation = cv2.warpAffine(img_translation, M, (iw, ih))

    rad = angle * (math.pi / 180.0)

    return img_rotation, rad


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
          o_points = self.get_landmark_points(faceLms.landmark, ih, iw)
          input_image, points = self.resize_align(input_image, o_points, self.size)
          
          ih, iw, ic = input_image.shape
          img, rad = self.GetRadian(input_image, points[4], points[8])

          for id, lm in enumerate(points):
            p = self.RotatePoint(np.array((int(iw/2), int(ih/2))), [lm[0] - self.anchorX , lm[1] - self.anchorY], -rad)
            (x, y) = int(p[0]), int(p[1])
            cv2.circle(img, (x,y),3,(255,0,0),3)
            
            if id in self._landmarks.FACE_CONTOUR : input_Face_contour.append((x,y))
            if id in self._landmarks.LEFT_EYE : 
              input_left_eye.append((x, y))                
              if id in self._landmarks.TRANSFORM_LEFT_EYE : transform_input_left_eye.append((x, y))
              
            if id in self._landmarks.RIGHT_EYE : 
              input_right_eye.append((x, y))
              if id in self._landmarks.TRANSFORM_RIGHT_EYE : transform_input_right_eye.append((x, y))

            if id in self._landmarks.LEFT_EYE_B : 
              input_left_eye_b.append((x, y))
              if id in self._landmarks.TRANSFORM_LEFT_EYE_B : transform_input_left_eye_b.append((x, y))
              
            if id in self._landmarks.RIGHT_EYE_B : 
              input_right_eye_b.append((x, y))
              if id in self._landmarks.TRANSFORM_RIGHT_EYE_B : transform_input_right_eye_b.append((x, y))

            if id in self._landmarks.NOSE : 
              input_nose.append((x, y))
              if id in self._landmarks.TRANSFORM_NOSE : transform_input_nose.append((x, y))

            if id in self._landmarks.MOUTH : 
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

    Angle, h_scale, v_scale, h_trans, v_trans = self.get_transform(transform_input_nose, self._landmarks.Asset_transform_nose, Nose_ID, 'NOSE')
    self.value_to_list(Nose, 0, h_scale, v_scale, 0, v_trans)
    
    Angle_l, h_scale_l, v_scale_l, h_trans_l, v_trans_l = self.get_transform(transform_input_left_eye, self._landmarks.Asset_transform_left_eyes, Eye_ID, 'LEFT_EYE')
    Angle_r, h_scale_r, v_scale_r, h_trans_r, v_trans_r = self.get_transform(transform_input_right_eye, self._landmarks.Asset_transform_right_eyes, Eye_ID, 'RIGHT_EYE')

    if Angle_l < Angle_r:
      self.value_to_list(L_Eye, Angle_l, h_scale_l, v_scale_l, h_trans_l, v_trans_l)
      self.value_to_list(R_Eye, -Angle_l, h_scale_l, v_scale_l, -h_trans_l, v_trans_l)
    else:
      self.value_to_list(L_Eye, -Angle_r, h_scale_r, v_scale_r, -h_trans_r, v_trans_r)
      self.value_to_list(R_Eye, Angle_r, h_scale_r, v_scale_r, h_trans_r, v_trans_r)

    Angle_l, h_scale_l, v_scale_l, h_trans_l, v_trans_l = self.get_transform(transform_input_left_eye_b, self._landmarks.Asset_transform_left_eyes_b, Eye_B_ID, 'LEFT_EYE_B')
    Angle_r, h_scale_r, v_scale_r, h_trans_r, v_trans_r = self.get_transform(transform_input_right_eye_b, self._landmarks.Asset_transform_right_eyes_b, Eye_B_ID, 'RIGHT_EYE_B')

    if Angle_l < Angle_r:
      self.value_to_list(R_Eye_b, Angle_l, h_scale_l, v_scale_l, h_trans_l, v_trans_l)
      self.value_to_list(L_Eye_b, -Angle_l, h_scale_l, v_scale_l, -h_trans_l, v_trans_l)
    else:
      self.value_to_list(R_Eye_b, -Angle_r, h_scale_r, v_scale_r, -h_trans_r, v_trans_r)
      self.value_to_list(L_Eye_b, Angle_r, h_scale_r, v_scale_r, h_trans_r, v_trans_r)


    Angle, h_scale, v_scale, h_trans, v_trans  = self.get_transform(transform_input_mouth, self._landmarks.Asset_transform_mouths, Mouth_ID, 'MOUTH')
    self.value_to_list(Mouth, Angle, h_scale, v_scale, 0, v_trans)
    
    transform_ = (Face_contour, Nose, L_Eye, R_Eye, L_Eye_b, R_Eye_b, Mouth)
    return [Face_contour_ID, Nose_ID, Eye_ID,  Eye_ID, Eye_B_ID, Eye_B_ID, Mouth_ID], transform_




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
      for faceLms in results.multi_face_landmarks:
          o_points = self.get_landmark_points(faceLms.landmark, ih, iw)
          input_image, points = self.resize_align(input_image, o_points, self.size)
          
          ih, iw, ic = input_image.shape
          img, rad = self.GetRadian(input_image, points[4], points[8])

          for id, lm in enumerate(points):
            p = self.RotatePoint(np.array((int(iw/2), int(ih/2))), [lm[0] - self.anchorX , lm[1] - self.anchorY], -rad)
            (x, y) = int(p[0]), int(p[1])
            cv2.circle(img, (x,y),3,(255,0,0),3)
            
            if id in self._landmarks.FACE_CONTOUR : input_Face_contour.append((x,y))
            if id in self._landmarks.LEFT_EYE : 
              input_left_eye.append((x, y))                
              if id in self._landmarks.TRANSFORM_LEFT_EYE : transform_input_left_eye.append((x, y))
              
            if id in self._landmarks.RIGHT_EYE : 
              input_right_eye.append((x, y))
              if id in self._landmarks.TRANSFORM_RIGHT_EYE : transform_input_right_eye.append((x, y))

            if id in self._landmarks.LEFT_EYE_B : 
              input_left_eye_b.append((x, y))
              if id in self._landmarks.TRANSFORM_LEFT_EYE_B : transform_input_left_eye_b.append((x, y))
              
            if id in self._landmarks.RIGHT_EYE_B : 
              input_right_eye_b.append((x, y))
              if id in self._landmarks.TRANSFORM_RIGHT_EYE_B : transform_input_right_eye_b.append((x, y))

            if id in self._landmarks.NOSE : 
              input_nose.append((x, y))
              if id in self._landmarks.TRANSFORM_NOSE : transform_input_nose.append((x, y))

            if id in self._landmarks.MOUTH : 
              input_mouth.append((x, y))
              if id in self._landmarks.TRANSFORM_MOUTH : transform_input_mouth.append((x, y))
          cv2.imshow("sss",img)

    for point in transform_input_left_eye:
      cv2.circle(input_image,point,3,(255,0,0),2)
    print(f'TRANSFORM_Asset_{id_data}_LEFT_EYE =',transform_input_left_eye)

    for point in transform_input_right_eye:
      cv2.circle(input_image,point,3,(255,0,0),2)
    print(f'TRANSFORM_Asset_{id_data}_RIGHT_EYE =',transform_input_right_eye)
    
    for point in transform_input_left_eye_b:
      cv2.circle(input_image,point,3,(255,0,0),2)
    print(f'TRANSFORM_Asset_{id_data}_LEFT_EYE_B =',transform_input_left_eye_b)
      
    for point in transform_input_right_eye_b:
      cv2.circle(input_image,point,3,(255,0,0),2)
    print(f'TRANSFORM_Asset_{id_data}_RIGHT_EYE_B =',transform_input_right_eye_b)

    for point in transform_input_nose:
      cv2.circle(input_image,point,3,(255,0,0),2)
    print(f'TRANSFORM_Asset_{id_data}_NOSE =',transform_input_nose)
      
    for point in transform_input_mouth:
      cv2.circle(input_image,point,3,(255,0,0),2)
    print(f'TRANSFORM_Asset_{id_data}_MOUTH =',transform_input_mouth)
      

    cv2.imwrite("result.jpg",input_image)



if __name__ == "__main__":
  id = 1
  input_image = cv2.imread(f"{id}.png")
  #input_image = cv2.imread("1.png")

  land = LANDMARK_MATCHING()
  input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  
  land.makeDatabase(input_image, id)
  land.landmark_part_matching(input_image)