import os
import cv2
import numpy as np
import json
import time

from typing import Optional,List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from face_matcher import LANDMARK_MATCHING



class PartInfo(BaseModel):
    type: int 
    asset_id :int
    rotation:float
    h_scale:float
    v_scale:float
    h_trans:float
    v_trans:float



app = FastAPI()
detector = LANDMARK_MATCHING()



async def process(img):
    # Process landmark detection.
    res, transform  = detector.landmark_part_matching(img)

    # Return if no face detected.
    if len(res) == 0:
        return {'status':'fail','message':'No face detected'}

    # Make json result
    # TODO add rotation, scale, translation value
    data = []
    for i, num in enumerate(res):
        data.append({"type":int(i), "asset_id":int(num), "rotation":transform[i][0], "h_scale":transform[i][1], "v_scale":transform[i][2], "h_trans":transform[i][3], "v_trans":transform[i][4]})
        #data.append({"type":int(i), "asset_id":int(num), "rotation":transform[i][0], "h_scale":transform[i][1], "v_scale":transform[i][2], "h_trans":0, "v_trans":0})

    return {'status': 'success', 'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]) , 'data': json.dumps(data)}


@app.post("/processFace")
async def processFace(image: bytes = File(...)):
    start = time.time()

    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)

    ret = await process(img)

    print('time: ' + str(time.time()-start) + ' ')
    return ret




@app.post("/finalize")
async def finalize(data:List[PartInfo]):
    for d in data:
        print(f'{d.type} {d.asset_id} {d.rotation} {d.h_scale} {d.v_scale} {d.h_trans} {d.v_trans}')

    # TODO store finalized values
    


    return {'status':'success'}
