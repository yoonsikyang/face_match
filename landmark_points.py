import numpy as np

class LANDMARK_points:
  def __init__(self):

    self.Asset_size = 2

    # Total Landmark
    ############################################################################################################################################################################################################
    self.FACE_CONTOUR = [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 400, 454]
    self.LEFT_EYE = [249, 252, 253, 254, 255, 256, 257, 258, 259, 260, 263, 286, 339, 341, 359, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 414, 463, 466, 467]
    self.RIGHT_EYE = [7, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 56, 110, 112, 130, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 190, 243, 246, 247]
    self.LEFT_EYE_B = [276, 283, 282, 285, 293, 295, 296, 300, 334, 336] 
    self.RIGHT_EYE_B = [46, 52, 53, 55, 63, 65, 66, 70, 105, 107]
    self.NOSE = [1, 2, 3, 4, 5, 6, 8, 19, 20, 44, 45, 51, 59, 60, 64, 75, 79, 94, 97, 98, 99, 115, 122, 125, 131, 134, 141, 166, 168, 174, 195, 196, 197, 218, 219, 220, 235, 236, 237, 238, 239, 240, 241, 242, 248, 250, 274, 275, 281, 289, 290, 305, 309, 344, 351, 354, 360, 363, 370, 392, 399, 419, 440, 456, 438,  457, 458, 459,  461, 462]
    self.MOUTH = [0, 11, 12, 13, 14, 15, 16, 17, 37, 38, 39, 40, 41, 42, 61, 62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 95, 96, 146, 178, 179, 180, 181, 183, 184, 185, 191, 267, 268, 269, 270, 271, 272, 291, 292, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415, 302, 303, 304,  306, 307, 315, 316, 319, 320, 325, 403, 404, 407, 408 ]
    
    # Asset 0
    Asset_0_FACE_CONTOUR = [(254, 75), (73, 147), (88, 117), (99, 377), (156, 81), (77, 293), (114, 95), (202, 75), (66, 217), (86, 334), (135, 437), (222, 497), (177, 474), (158, 459), (254, 500), (66, 178), (116, 411), (198, 487), (71, 255), (428, 150), (414, 121), (404, 378), (349, 83), (425, 295), (389, 98), (306, 76), (435, 219), (416, 336), (370, 437), (285, 497), (329, 474), (348, 459), (434, 181), (387, 412), (309, 487), (431, 257)]
    Asset_0_LEFT_EYE = [(367, 221), (322, 236), (339, 237), (354, 237), (373, 227), (308, 233), (346, 185), (327, 187), (364, 188), (375, 196), (372, 216), (310, 196), (366, 234), (299, 229), (379, 217), (298, 223), (352, 226), (339, 227), (325, 227), (311, 225), (302, 225), (313, 210), (328, 204), (342, 203), (356, 206), (365, 210), (361, 224), (303, 218), (296, 211), (292, 224), (369, 214), (381, 205)]
    Asset_0_RIGHT_EYE =  [(138, 217), (183, 235), (167, 235), (152, 235), (131, 224), (197, 232), (159, 184), (178, 186), (141, 187), (130, 194), (132, 212), (194, 195), (139, 231), (206, 228), (125, 213), (206, 222), (153, 224), (166, 225), (180, 225), (193, 223), (202, 223), (191, 208), (176, 202), (162, 200), (148, 203), (140, 207), (144, 221), (201, 217), (208, 210), (212, 223), (136, 210), (124, 202)]
    Asset_0_LEFT_EYE_B = [(397, 180), (361, 160), (382, 166), (288, 177), (390, 155), (330, 162), (333, 147), (405, 172), (366, 146), (293, 153)]
    Asset_0_RIGHT_EYE_B = [(106, 179), (144, 160), (121, 166), (218, 178), (112, 154), (176, 163), (173, 148), (97, 170), (138, 146), (213, 154)]
    Asset_0_NOSE = [(253, 329), (253, 343), (240, 271), (254, 312), (253, 289), (253, 229), (253, 184), (253, 337), (239, 335), (241, 329), (239, 312), (239, 290), (219, 331), (230, 336), (207, 329), (222, 334), (225, 326), (253, 340), (235, 342), (212, 338), (233, 339), (215, 316), (238, 233), (246, 336), (215, 304), (226, 295), (247, 339), (219, 329), (253, 206), (228, 261), (253, 268), (239, 253), (253, 249), (220, 324), (212, 327), (226, 313), (213, 332), (228, 278), (230, 325), (238, 333), (231, 328), (217, 336), (241, 335), (243, 338), (266, 271), (267, 336), (266, 329), (268, 312), (267, 290), (287, 331), (276, 336), (283, 334), (281, 327), (291, 316), (268, 233), (261, 336), (291, 305), (280, 296), (259, 339), (287, 329), (278, 261), (267, 253), (286, 324), (280, 314), (278, 278), (276, 326), (268, 333), (276, 328), (265, 335), (263, 338)]
    Asset_0_MOUTH = [(253, 374), (253, 382), (253, 392), (253, 398), (253, 399), (253, 408), (253, 419), (253, 431), (236, 370), (238, 391), (218, 378), (206, 386), (224, 393), (214, 395), (194, 398), (198, 398), (236, 381), (221, 386), (210, 390), (196, 398), (202, 402), (200, 398), (217, 398), (227, 397), (239, 397), (235, 429), (236, 418), (237, 407), (239, 398), (216, 398), (213, 402), (209, 407), (207, 413), (209, 398), (205, 400), (199, 405), (227, 398), (223, 404), (221, 413), (219, 423), (205, 396), (202, 394), (198, 392), (209, 398), (271, 371), (269, 392), (289, 379), (300, 387), (282, 394), (293, 396), (313, 400), (309, 400), (270, 382), (286, 387), (297, 392), (311, 400), (304, 404), (307, 400), (290, 399), (280, 398), (267, 398), (271, 429), (270, 418), (269, 407), (267, 398), (290, 399), (293, 403), (297, 408), (299, 414), (298, 399), (301, 401), (307, 407), (280, 398), (282, 405), (285, 413), (287, 423), (301, 398), (305, 396), (308, 394), (298, 399)]
    
    # Asset 1
    Asset_1_FACE_CONTOUR = [(251, 67), (69, 141), (83, 111), (97, 374), (151, 74), (75, 289), (109, 88), (197, 67), (64, 213), (84, 331), (132, 435), (220, 496), (176, 473), (156, 458), (251, 499), (64, 173), (113, 409), (196, 487), (69, 250), (428, 142), (415, 112), (404, 374), (349, 74), (424, 290), (390, 89), (304, 68), (434, 213), (416, 331), (369, 434), (282, 496), (326, 472), (346, 457), (434, 174), (388, 408), (306, 486), (430, 251)]
    Asset_1_LEFT_EYE = [(368, 218), (323, 237), (339, 237), (354, 235), (374, 224), (309, 234), (347, 182), (327, 185), (365, 185), (376, 193), (373, 213), (311, 194), (367, 231), (300, 230), (381, 213), (299, 224), (353, 225), (339, 227), (326, 227), (313, 226), (304, 225), (314, 208), (328, 200), (343, 198), (357, 201), (365, 206), (362, 222), (304, 218), (297, 211), (294, 225), (370, 209), (382, 202)]
    Asset_1_RIGHT_EYE = [(133, 219), (180, 237), (163, 237), (147, 236), (127, 225), (193, 233), (153, 181), (173, 184), (135, 185), (124, 193), (128, 213), (190, 193), (134, 232), (202, 230), (120, 214), (203, 223), (148, 225), (162, 227), (176, 227), (190, 225), (198, 225), (188, 206), (173, 199), (158, 197), (144, 200), (135, 205), (139, 222), (198, 217), (204, 210), (208, 224), (131, 210), (119, 202)]
    Asset_1_LEFT_EYE_B = [(399, 176), (362, 158), (384, 164), (287, 175), (393, 151), (330, 160), (333, 145), (407, 167), (367, 143), (292, 151)]
    Asset_1_RIGHT_EYE_B = [(100, 177), (137, 158), (115, 164), (211, 176), (105, 152), (168, 160), (165, 146), (91, 168), (130, 144), (205, 151)]
    Asset_1_NOSE = [(250, 333), (250, 346), (236, 272), (250, 315), (250, 291), (250, 230), (250, 181), (250, 340), (235, 338), (237, 332), (235, 315), (235, 292), (214, 332), (226, 337), (202, 329), (217, 335), (220, 328), (250, 343), (230, 344), (207, 338), (228, 341), (211, 318), (234, 233), (242, 339), (210, 306), (222, 297), (244, 342), (214, 330), (250, 205), (224, 262), (250, 270), (235, 254), (250, 251), (216, 326), (207, 328), (222, 316), (207, 332), (224, 279), (226, 328), (234, 336), (226, 330), (212, 336), (237, 338), (239, 341), (263, 272), (264, 338), (263, 332), (265, 315), (264, 292), (286, 332), (274, 337), (282, 335), (279, 328), (289, 318), (265, 233), (257, 339), (289, 306), (277, 297), (256, 342), (286, 330), (275, 262), (264, 254), (284, 326), (278, 316), (275, 279), (273, 328), (265, 336), (273, 330), (262, 338), (260, 341)]
    Asset_1_MOUTH = [(251, 384), (251, 390), (250, 397), (250, 401), (250, 401), (250, 409), (250, 419), (250, 429), (232, 380), (234, 395), (215, 384), (203, 389), (220, 395), (209, 395), (187, 394), (191, 395), (233, 388), (217, 390), (206, 392), (189, 394), (196, 399), (193, 395), (211, 397), (222, 398), (235, 400), (231, 426), (232, 417), (233, 407), (235, 400), (211, 397), (208, 400), (205, 404), (202, 409), (203, 396), (200, 397), (194, 401), (222, 398), (219, 404), (217, 411), (215, 419), (200, 395), (197, 393), (194, 392), (203, 396), (269, 380), (267, 396), (286, 385), (297, 389), (280, 395), (291, 396), (312, 394), (309, 395), (268, 389), (283, 391), (294, 392), (310, 395), (303, 399), (307, 395), (289, 398), (278, 399), (265, 400), (270, 427), (269, 417), (267, 408), (265, 400), (289, 397), (292, 401), (295, 405), (298, 410), (297, 397), (300, 398), (306, 401), (278, 399), (281, 404), (284, 411), (286, 419), (300, 395), (303, 394), (305, 392), (297, 397)]
    
    # Transform Landmark
    ############################################################################################################################################################################################################
    self.TRANSFORM_LEFT_EYE = [ 463, 257, 359, 253]      
    self.TRANSFORM_RIGHT_EYE = [130, 27, 155, 23]  
    self.TRANSFORM_LEFT_EYE_B = [296, 334, 282, 295]   
    self.TRANSFORM_RIGHT_EYE_B = [105, 66, 65, 52]    
    self.TRANSFORM_NOSE = [131, 8, 360, 141] 
    self.TRANSFORM_MOUTH = [78, 0, 291, 17]

    self.POINTS_ROTATE = [4, 175, 33, 263, 61, 291]

        
    # Asset 0
    TRANSFORM_Asset_0_LEFT_EYE = [(328, 229), (335, 180), (367, 209), (286, 216)]
    TRANSFORM_Asset_0_RIGHT_EYE = [(166, 227), (159, 178), (128, 206), (199, 216)]
    TRANSFORM_Asset_0_LEFT_EYE_B = [(349, 155), (319, 157), (322, 144), (353, 142)]
    TRANSFORM_Asset_0_RIGHT_EYE_B = [(146, 156), (175, 158), (173, 144), (140, 141)]
    TRANSFORM_Asset_0_NOSE = [(246, 177), (210, 293), (241, 326), (282, 294)]
    TRANSFORM_Asset_0_MOUTH = [(248, 358), (248, 411), (198, 381), (304, 383)]
 
    # Asset 1
    TRANSFORM_Asset_1_LEFT_EYE = [(332, 226), (339, 176), (371, 203), (290, 215)]
    TRANSFORM_Asset_1_RIGHT_EYE = [(170, 227), (161, 176), (130, 205), (202, 215)]
    TRANSFORM_Asset_1_LEFT_EYE_B = [(353, 154), (323, 155), (326, 141), (358, 141)]
    TRANSFORM_Asset_1_RIGHT_EYE_B = [(145, 155), (174, 157), (170, 143), (139, 142)]
    TRANSFORM_Asset_1_NOSE = [(250, 175), (213, 290), (245, 324), (286, 289)]
    TRANSFORM_Asset_1_MOUTH = [(251, 362), (251, 403), (198, 373), (309, 371)]


    ############################################################################################################################################################################################################

    # self.input_Face_contour = []
    # self.input_left_eye = []
    # self.input_right_eye = []
    # self.input_left_eye_b = []
    # self.input_right_eye_b = []
    # self.input_nose = []
    # self.input_mouth = []
     
    # self.transform_input_Face_contour = []
    # self.transform_input_left_eye = []
    # self.transform_input_right_eye = []
    # self.transform_input_left_eye_b = []
    # self.transform_input_right_eye_b = []
    # self.transform_input_nose = []
    # self.transform_input_mouth = []

    self.Asset_Face_contours = []
    self.Asset_left_eyes = []
    self.Asset_right_eyes = []
    self.Asset_left_eyes_b = []
    self.Asset_right_eyes_b = []
    self.Asset_nose = []
    self.Asset_mouths = []

    self.Asset_transform_left_eyes = []
    self.Asset_transform_right_eyes = []
    self.Asset_transform_left_eyes_b = []
    self.Asset_transform_right_eyes_b = []
    self.Asset_transform_nose = []
    self.Asset_transform_mouths = []

    self.Face_contour=[]
    self.Nose=[]
    self.L_Eye = []
    self.R_Eye = []
    self.L_Eye_b = []
    self.R_Eye_b = []
    self.Mouth = []
    
    for i in range(self.Asset_size):
      # Total landmark
      _asset_ = "Asset_{}_FACE_CONTOUR".format(i)
      self.Asset_Face_contours.append(locals()[_asset_])
      _asset_ = "Asset_{}_LEFT_EYE".format(i)
      self.Asset_left_eyes.append(locals()[_asset_])
      _asset_ = "Asset_{}_RIGHT_EYE".format(i)
      self.Asset_right_eyes.append(locals()[_asset_])
      _asset_ = "Asset_{}_LEFT_EYE_B".format(i)
      self.Asset_left_eyes_b.append(locals()[_asset_])
      _asset_ = "Asset_{}_RIGHT_EYE_B".format(i)
      self.Asset_right_eyes_b.append(locals()[_asset_])
      _asset_ = "Asset_{}_NOSE".format(i)
      self.Asset_nose.append(locals()[_asset_])
      _asset_ = "Asset_{}_MOUTH".format(i)
      self.Asset_mouths.append(locals()[_asset_])

      # Transform landmark
      _asset_ = "TRANSFORM_Asset_{}_LEFT_EYE".format(i)
      self.Asset_transform_left_eyes.append(locals()[_asset_])
      _asset_ = "TRANSFORM_Asset_{}_RIGHT_EYE".format(i)
      self.Asset_transform_right_eyes.append(locals()[_asset_])
      _asset_ = "TRANSFORM_Asset_{}_LEFT_EYE_B".format(i)
      self.Asset_transform_left_eyes_b.append(locals()[_asset_])
      _asset_ = "TRANSFORM_Asset_{}_RIGHT_EYE_B".format(i)
      self.Asset_transform_right_eyes_b.append(locals()[_asset_])
      _asset_ = "TRANSFORM_Asset_{}_NOSE".format(i)
      self.Asset_transform_nose.append(locals()[_asset_])
      _asset_ = "TRANSFORM_Asset_{}_MOUTH".format(i)
      self.Asset_transform_mouths.append(locals()[_asset_])

    
