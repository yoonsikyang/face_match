import numpy as np
import csv  


class LANDMARK_points:
  def __init__(self):

    # 해당 ID의 file 명 정의
    # WOMAN
    #self.WOMAN_FACE_NAME = ['SHAPE_W_AT10_W10_W10_2']
    #self.WOMAN_FACE_NAME = ['SHAPE_W_AT10_W10_W10_2', 'SHAPE_W_DM10_W10_W10_2', 'SHAPE_W_LO10_W10_W10_2', 'SHAPE_W_LR10_W10_W10_2', 'SHAPE_W_LS10_W10_W10_2', 'SHAPE_W_LV10_W10_W10_2', 'SHAPE_W_OV10_W10_W10_2', 'SHAPE_W_RO10_W10_W10_2',  'SHAPE_W_SQ10_W10_W10_2', 'SHAPE_W_VT10_W10_W10_2']
    #self.WOMAN_EYE_NAME = ['EYE_W_AU10_LCS', 'EYE_W_AU20_LCS', 'EYE_W_AU30_LCS', 'EYE_W_AU40_LCS', 'EYE_W_AU50_LCS', 'EYE_W_BE10_LCS', 'EYE_W_BE20_LCS', 'EYE_W_BE30_LCS', 'EYE_W_BE40_LCS', 'EYE_W_BE50_LCS', 'EYE_W_BE60_LCS', 'EYE_W_BE70_LCS', 'EYE_W_BE80_LCS', 'EYE_W_BE90_LCS', 'EYE_W_BF10_LCS', 'EYE_W_BF20_LCS', 'EYE_W_BF30_LCS', 'EYE_W_BF40_LCS', 'EYE_W_BF50_LCS', 'EYE_W_BF60_LCS', 'EYE_W_BF70_LCS', 'EYE_W_LE10_LCS', 'EYE_W_LE20_LCS', 'EYE_W_LE30_LCS', 'EYE_W_LE40_LCS', 'EYE_W_LE50_LCS', 'EYE_W_LE60_LCS', 'EYE_W_LE70_LCS', 'EYE_W_LE80_LCS', 'EYE_W_LE90_LCS', 'EYE_W_ND10_LCS', 'EYE_W_ND20_LCS', 'EYE_W_ND30_LCS', 'EYE_W_ND40_LCS', 'EYE_W_ND50_LCS', 'EYE_W_ND60_LCS', 'EYE_W_ND70_LCS', 'EYE_W_ND80_LCS', 'EYE_W_RE10_LCS', 'EYE_W_RE20_LCS', 'EYE_W_RE30_LCS', 'EYE_W_RE40_LCS', 'EYE_W_RE50_LCS']
    #self.WOMAN_EYE_BROW_NAME = ['BROW_W_FL06', 'BROW_W_HA06', 'BROW_W_LW06', 'BROW_W_RH06', 'BROW_W_RL06', 'BROW_W_RM06', 'BROW_W_SF06', 'BROW_W_SH06', 'BROW_W_SL06', 'BROW_W_SM06', 'BROW_W_SS06', 'BROW_W_ST06']
    #self.WOMAN_MOUTH_NAME = ['LIP_W_BW10', 'LIP_W_BW20', 'LIP_W_BW30', 'LIP_W_BW40', 'LIP_W_DL10', 'LIP_W_DL20', 'LIP_W_FL10', 'LIP_W_FL20', 'LIP_W_FL30', 'LIP_W_FL40', 'LIP_W_HL10', 'LIP_W_HL20', 'LIP_W_HL30', 'LIP_W_HL40', 'LIP_W_HL50', 'LIP_W_HL60', 'LIP_W_HS10', 'LIP_W_HS20', 'LIP_W_HS30', 'LIP_W_HS40', 'LIP_W_HS50', 'LIP_W_HU10', 'LIP_W_HU20', 'LIP_W_HU30', 'LIP_W_RO10', 'LIP_W_RO20', 'LIP_W_RO30', 'LIP_W_TN10', 'LIP_W_TN20', 'LIP_W_TN30', 'LIP_W_TN40', 'LIP_W_TN50']
    #self.WOMAN_NOSE_NAME = ['NOSE_W_WN02', 'NOSE_W_WN04', 'NOSE_W_WN06', 'NOSE_W_WN08', 'NOSE_W_WN10', 'NOSE_W_WN12', 'NOSE_W_WN14', 'NOSE_W_WN16', 'NOSE_W_WN18', 'NOSE_W_WN20', 'NOSE_W_WN22', 'NOSE_W_WN24', 'NOSE_W_WN26', 'NOSE_W_WN28', 'NOSE_W_WN30', 'NOSE_W_WN32', 'NOSE_W_WN34', 'NOSE_W_WN36', 'NOSE_W_WN38', 'NOSE_W_WN40', 'NOSE_W_WN42', 'NOSE_W_WN44', 'NOSE_W_WN46']
    
    
    self.WOMAN_FACE_NAME = ['SHAPE_W_AT10']
    #self.WOMAN_FACE_NAME = ['SHAPE_W_AT10', 'SHAPE_W_DM10', 'SHAPE_W_LO10', 'SHAPE_W_LR10', 'SHAPE_W_LS10', 'SHAPE_W_LV10', 'SHAPE_W_OV10', 'SHAPE_W_RO10',  'SHAPE_W_SQ10', 'SHAPE_W_VT10']
    self.WOMAN_EYE_NAME = ['EYE_W_AU10', 'EYE_W_AU20', 'EYE_W_AU30', 'EYE_W_AU40', 'EYE_W_AU50', 'EYE_W_BE10', 'EYE_W_BE20', 'EYE_W_BE30', 'EYE_W_BE40', 'EYE_W_BE50', 'EYE_W_BE60', 'EYE_W_BE70', 'EYE_W_BE80', 'EYE_W_BE90', 'EYE_W_BF10', 'EYE_W_BF20', 'EYE_W_BF30', 'EYE_W_BF40', 'EYE_W_BF50', 'EYE_W_BF60', 'EYE_W_BF70', 'EYE_W_LE10', 'EYE_W_LE20', 'EYE_W_LE30', 'EYE_W_LE40', 'EYE_W_LE50', 'EYE_W_LE60', 'EYE_W_LE70', 'EYE_W_LE80', 'EYE_W_LE90', 'EYE_W_ND10', 'EYE_W_ND20', 'EYE_W_ND30', 'EYE_W_ND40', 'EYE_W_ND50', 'EYE_W_ND60', 'EYE_W_ND70', 'EYE_W_ND80', 'EYE_W_RE10', 'EYE_W_RE20', 'EYE_W_RE30', 'EYE_W_RE40', 'EYE_W_RE50']
    self.WOMAN_EYE_BROW_NAME = ['BROW_W_FL06', 'BROW_W_HA06', 'BROW_W_LW06', 'BROW_W_RH06', 'BROW_W_RL06', 'BROW_W_RM06', 'BROW_W_SF06', 'BROW_W_SH06', 'BROW_W_SL06', 'BROW_W_SM06', 'BROW_W_SS06', 'BROW_W_ST06']
    self.WOMAN_MOUTH_NAME = ['LIP_W_BW10', 'LIP_W_BW20', 'LIP_W_BW30', 'LIP_W_BW40', 'LIP_W_DL10', 'LIP_W_DL20', 'LIP_W_FL10', 'LIP_W_FL20', 'LIP_W_FL30', 'LIP_W_FL40', 'LIP_W_HL10', 'LIP_W_HL20', 'LIP_W_HL30', 'LIP_W_HL40', 'LIP_W_HL50', 'LIP_W_HL60', 'LIP_W_HS10', 'LIP_W_HS20', 'LIP_W_HS30', 'LIP_W_HS40', 'LIP_W_HS50', 'LIP_W_HU10', 'LIP_W_HU20', 'LIP_W_HU30', 'LIP_W_RO10', 'LIP_W_RO20', 'LIP_W_RO30', 'LIP_W_TN10', 'LIP_W_TN20', 'LIP_W_TN30', 'LIP_W_TN40', 'LIP_W_TN50']
    self.WOMAN_NOSE_NAME = ['NOSE_W_WN02', 'NOSE_W_WN04', 'NOSE_W_WN06', 'NOSE_W_WN08', 'NOSE_W_WN10', 'NOSE_W_WN12', 'NOSE_W_WN14', 'NOSE_W_WN16', 'NOSE_W_WN18', 'NOSE_W_WN20', 'NOSE_W_WN22', 'NOSE_W_WN24', 'NOSE_W_WN26', 'NOSE_W_WN28', 'NOSE_W_WN30', 'NOSE_W_WN32', 'NOSE_W_WN34', 'NOSE_W_WN36', 'NOSE_W_WN38', 'NOSE_W_WN40', 'NOSE_W_WN42', 'NOSE_W_WN44', 'NOSE_W_WN46']
    ################### man
    self.MAN_FACE_NAME = ['SHAPE_M_DM10']
    self.MAN_EYE_NAME = ['EYE_M_AU15', 'EYE_M_AU20', 'EYE_M_AU25', 'EYE_M_AU30', 'EYE_M_AU35', 'EYE_M_AU40', 'EYE_M_AU45', 'EYE_M_BE05', 'EYE_M_BE10', 'EYE_M_BE15', 'EYE_M_BE20', 'EYE_M_BE25', 'EYE_M_BE30', 'EYE_M_BE35', 'EYE_M_BE40', 'EYE_M_BE45', 'EYE_M_BE50', 'EYE_M_BE55', 'EYE_M_BE60', 'EYE_M_BE65', 'EYE_M_BE70', 'EYE_M_BE75', 'EYE_M_BF10', 'EYE_M_BF15', 'EYE_M_BF20', 'EYE_M_BF25', 'EYE_M_BF30', 'EYE_M_BF35', 'EYE_M_BF40', 'EYE_M_BF45', 'EYE_M_BF50', 'EYE_M_BF55', 'EYE_M_BF60', 'EYE_M_BF65', 'EYE_M_LE10', 'EYE_M_LE20', 'EYE_M_LE30', 'EYE_M_LE40', 'EYE_M_LE50', 'EYE_M_LE60', 'EYE_M_LE70', 'EYE_M_ND10', 'EYE_M_ND15', 'EYE_M_ND20', 'EYE_M_ND25', 'EYE_M_ND30', 'EYE_M_ND35', 'EYE_M_ND40', 'EYE_M_ND45', 'EYE_M_ND50', 'EYE_M_ND55', 'EYE_M_ND60', 'EYE_M_ND65', 'EYE_M_RE10', 'EYE_M_RE20', 'EYE_M_RE30', 'EYE_M_RE40', 'EYE_M_RE50', 'EYE_M_RE60']
    self.MAN_EYE_BROW_NAME = ['BROW_M_AF02', 'BROW_M_FL02', 'BROW_M_HA02', 'BROW_M_HL02', 'BROW_M_RA02', 'BROW_M_RF02', 'BROW_M_RH02', 'BROW_M_SA02', 'BROW_M_SF02', 'BROW_M_SH02', 'BROW_M_SL02']
    self.MAN_MOUTH_NAME = ['LIP_M_BW10', 'LIP_M_BW20', 'LIP_M_BW30', 'LIP_M_BW40', 'LIP_M_DL10', 'LIP_M_DL20', 'LIP_M_DL30', 'LIP_M_DL40', 'LIP_M_FL10', 'LIP_M_FL20', 'LIP_M_FL30', 'LIP_M_FL40', 'LIP_M_FL50', 'LIP_M_FL60', 'LIP_M_FL70', 'LIP_M_FL80', 'LIP_M_HL10', 'LIP_M_HL20', 'LIP_M_HL30', 'LIP_M_HL40', 'LIP_M_HS10', 'LIP_M_HS20', 'LIP_M_HS30', 'LIP_M_HU10', 'LIP_M_HU20', 'LIP_M_HU30', 'LIP_M_RL10', 'LIP_M_RL20', 'LIP_M_RL30', 'LIP_M_RL40', 'LIP_M_TN10', 'LIP_M_TN20', 'LIP_M_TN30', 'LIP_M_TN40', 'LIP_M_WL10', 'LIP_M_WL20', 'LIP_M_WL30', 'LIP_M_WL40']
    self.MAN_NOSE_NAME = ['NOSE_M_MN01', 'NOSE_M_MN02', 'NOSE_M_MN03', 'NOSE_M_MN04', 'NOSE_M_MN05', 'NOSE_M_MN06', 'NOSE_M_MN07', 'NOSE_M_MN08', 'NOSE_M_MN09', 'NOSE_M_MN10', 'NOSE_M_MN11', 'NOSE_M_MN12', 'NOSE_M_MN13', 'NOSE_M_MN14', 'NOSE_M_MN15', 'NOSE_M_MN16', 'NOSE_M_MN17', 'NOSE_M_MN18', 'NOSE_M_MN19', 'NOSE_M_MN20', 'NOSE_M_MN21', 'NOSE_M_MN22', 'NOSE_M_MN23', 'NOSE_M_MN24', 'NOSE_M_MN25', 'NOSE_M_MN26', 'NOSE_M_MN27', 'NOSE_M_MN28', 'NOSE_M_MN29', 'NOSE_M_MN30', 'NOSE_M_MN31']
    

     # Total Landmark
    ############################################################################################################################################################################################################
    self.FACE_CONTOUR = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454]
    self.LEFT_EYE = [249, 252, 253, 254, 255, 256, 257, 258, 259, 260, 263, 286, 339, 341, 359, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 414, 463, 466, 467]
    self.RIGHT_EYE = [7, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 56, 110, 112, 130, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 190, 243, 246, 247]
    self.LEFT_EYE_B = [276, 283, 282, 285, 293, 295, 296, 300, 334, 336] 
    self.RIGHT_EYE_B = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    self.NOSE = [1, 2, 3, 4, 5, 6, 8, 19, 20, 44, 45, 51, 59, 60, 64, 75, 79, 94, 97, 98, 99, 115, 122, 125, 131, 134, 141, 166, 168, 174, 195, 196, 197, 218, 219, 220, 235, 236, 237, 238, 239, 240, 241, 242, 248, 250, 274, 275, 281, 289, 290, 305, 309, 344, 351, 354, 360, 363, 370, 392, 399, 419, 440, 456, 438,  457, 458, 459,  461, 462]
    self.MOUTH = [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415, 302, 303, 304, 306, 307, 315, 316, 319, 320, 325, 403, 404, 407, 408 ]
    
        
    self.TRANSFORM_LEFT_EYE = [ 463, 257, 359, 253]
    self.TRANSFORM_RIGHT_EYE = [130, 27, 155, 23]
    self.TRANSFORM_LEFT_EYE_B = [296, 334, 282, 295]
    self.TRANSFORM_RIGHT_EYE_B = [105, 66, 65, 52]
    self.TRANSFORM_NOSE = [131, 8, 360, 141]
    self.TRANSFORM_MOUTH = [78, 0, 291, 17] #, 13, 14]
    

    # Woman   
    self.Asset_w_Face_contours = []
    self.Asset_w_left_eyes = []
    self.Asset_w_right_eyes = []
    self.Asset_w_left_eyes_b = []
    self.Asset_w_right_eyes_b = []
    self.Asset_w_nose = []
    self.Asset_w_mouths = []

    self.Asset_w_left_eyes_transform = []
    self.Asset_w_right_eyes_transform = []
    self.Asset_w_left_eyes_b_transform = []
    self.Asset_w_right_eyes_b_transform = []
    self.Asset_w_nose_transform = []
    self.Asset_w_mouths_transform = []


    # Man
    self.Asset_m_Face_contours = []
    self.Asset_m_left_eyes = []
    self.Asset_m_right_eyes = []
    self.Asset_m_left_eyes_b = []
    self.Asset_m_right_eyes_b = []
    self.Asset_m_nose = []
    self.Asset_m_mouths = []

    self.Asset_m_left_eyes_transform = []
    self.Asset_m_right_eyes_transform = []
    self.Asset_m_left_eyes_b_transform = []
    self.Asset_m_right_eyes_b_transform = []
    self.Asset_m_nose_transform = []
    self.Asset_m_mouths_transform = []




    self.Face_contour_load('FACE_CONTOUR')

    self.Facial_feature_load('LEFT_EYE')
    self.Facial_feature_load('RIGHT_EYE')
    self.Facial_feature_load('LEFT_EYE_BROW')
    self.Facial_feature_load('RIGHT_EYE_BROW')
    self.Facial_feature_load('NOSE')
    self.Facial_feature_load('LIP')

    self.Transform_load('TRANSFORM_LEFT_EYE')
    self.Transform_load('TRANSFORM_RIGHT_EYE')
    self.Transform_load('TRANSFORM_LEFT_EYE_BROW')
    self.Transform_load('TRANSFORM_RIGHT_EYE_BROW')
    self.Transform_load('TRANSFORM_NOSE')
    self.Transform_load('TRANSFORM_LIP')



  # woman 0 man 1
  def Transform_load(self, file):
    file_w = 'W_' + file
    with open('data/' + file_w + '.csv', 'rU') as data:
      reader = csv.reader(data)
      for row in reader:
        if file == 'TRANSFORM_LEFT_EYE' : self.Asset_w_left_eyes_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_RIGHT_EYE' : self.Asset_w_right_eyes_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_LEFT_EYE_BROW' : self.Asset_w_left_eyes_b_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_RIGHT_EYE_BROW' : self.Asset_w_right_eyes_b_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_NOSE' : self.Asset_w_nose_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_LIP' : self.Asset_w_mouths_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])

    file_m = 'M_' + file
    with open('data/' + file_m + '.csv', 'rU') as data:
      reader = csv.reader(data)
      for row in reader:
        if file == 'TRANSFORM_LEFT_EYE' : self.Asset_m_left_eyes_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_RIGHT_EYE' : self.Asset_m_right_eyes_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_LEFT_EYE_BROW' : self.Asset_m_left_eyes_b_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_RIGHT_EYE_BROW' : self.Asset_m_right_eyes_b_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_NOSE' : self.Asset_m_nose_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        elif file == 'TRANSFORM_LIP' : self.Asset_m_mouths_transform.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])

  def Facial_feature_load(self, file):
    file_w = 'W_' + file
    with open('data/' + file_w + '.csv', 'rU') as data:
      reader = csv.reader(data)
      for row in reader:
        test = []
        for rr in range(len(row)):
          temp = row[rr].split(",")
          temp[0] = temp[0].split("(")[1]
          temp[1] = temp[1].split(")")[0]
          test.append((float(temp[0]), float(temp[1])))
        if file == 'LEFT_EYE' : self.Asset_w_left_eyes.append(test)
        elif file == 'RIGHT_EYE' : self.Asset_w_right_eyes.append(test)
        elif file == 'LEFT_EYE_BROW' : self.Asset_w_left_eyes_b.append(test)
        elif file == 'RIGHT_EYE_BROW' : self.Asset_w_right_eyes_b.append(test)
        elif file == 'NOSE' : self.Asset_w_nose.append(test)
        elif file == 'LIP' : self.Asset_w_mouths.append(test)

    file_m = 'M_' + file
    with open('data/' + file_m + '.csv', 'rU') as data:
      reader = csv.reader(data)
      for row in reader:
        test = []
        for rr in range(len(row)):
          temp = row[rr].split(",")
          temp[0] = temp[0].split("(")[1]
          temp[1] = temp[1].split(")")[0]
          test.append((float(temp[0]), float(temp[1])))
        if file == 'LEFT_EYE' : self.Asset_m_left_eyes.append(test)
        elif file == 'RIGHT_EYE' : self.Asset_m_right_eyes.append(test)
        elif file == 'LEFT_EYE_BROW' : self.Asset_m_left_eyes_b.append(test)
        elif file == 'RIGHT_EYE_BROW' : self.Asset_m_right_eyes_b.append(test)
        elif file == 'NOSE' : self.Asset_m_nose.append(test)
        elif file == 'LIP' : self.Asset_m_mouths.append(test)

  def Face_contour_load(self, file):
    file_w = 'W_' + file
    with open('data/' + file_w + '.csv', 'rU') as data:
      reader = csv.reader(data)
      for row in reader:
        self.Asset_w_Face_contours.append((float(row[0]), float(row[1])))
        
        
    file_m = 'M_' + file
    with open('data/' + file_m + '.csv', 'rU') as data:
      reader = csv.reader(data)
      for row in reader:
        self.Asset_m_Face_contours.append((float(row[0]), float(row[1])))



  
  
    
