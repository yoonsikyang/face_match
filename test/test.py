from typing import Optional

from fastapi import FastAPI, File, UploadFile

import cv2
import numpy as np
import json

from face_matcher import LANDMARK_MATCHING

detector = LANDMARK_MATCHING()

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


async def process(img):
    # Process landmark detection.
    res = detector.landmark_part_matching(img)

    # Return if no face detected.
    if len(res) == 0:
      return False, {'status':'fail','message':'No face detected'}


    # Make json result
    # TODO add rotation, scale, translation value
    data = []
    for i, num in enumerate(res):
        data.append({"type":int(i), "asset_id":int(num), "rotation":0, "h_scale":1.0, "v_scale":1.0, "h_trans":0, "v_trans":0})
    
    return True, data


@app.post("/processFace")
async def processFace(image: bytes = File(...)):
    contents = await image.read() 
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_UNCHANGED)
    
    b_done, ret = process(img)

    if b_done:
        return {'status': 'success', 'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]) , 'data': json.dumps(ret)}
    else:    
        return ret
    
        
