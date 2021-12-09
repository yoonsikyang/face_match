from flask import Flask, request, render_template, Response, make_response

import cv2
import numpy as np
import json

from face_matcher import LANDMARK_MATCHING

detector = LANDMARK_MATCHING()

app = Flask(__name__)


@app.route('/processFace', methods=['POST'])
def processFace():
    r = request

    # Get image from the post request.
    f = request.files['image']
    img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #cv2.imwrite('test.png',img)

    # Process landmark detection.
    res = detector.landmark_part_matching(img)

    # Return if no face detected.
    if len(res) == 0:
      return {'status':'fail','message':'No face detected'}


    # Make json result
    # TODO add rotation, scale, translation value
    data = []
    for i, num in enumerate(res):
        data.append({"type":int(i), "asset_id":int(num), "rotation":0, "h_scale":1.0, "v_scale":1.0, "h_trans":0, "v_trans":0})

    # Return status
    return {'status': 'success', 'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]) , 'data': json.dumps(data)}



@app.route('/finalize', methods=['POST'])
def finalize():
    json_data = request.get_json()
    print(json_data)
    res = []
    # Parse json from device
    for data in json_data:
        res.append({"type":data["type"], "asset_id":data["asset_id"], "rotation":data["rotation"], "h_scale":data["h_scale"], "v_scale":data["v_scale"], "h_trans":data["h_trans"], "v_trans":data["v_trans"]})

    # TODO store finalized values
    


    return {'status':'success'}




if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        pass

    exit()


