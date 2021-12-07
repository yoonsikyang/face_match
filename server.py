from flask import Flask, request, render_template, Response, make_response

import cv2
import numpy as np

from face_matcher import LANDMARK_MATCHING

detector = LANDMARK_MATCHING()

app = Flask(__name__)


@app.route('/processFace', methods=['POST'])
def processFace():
    r = request
    
    # Get image from the post request.
    f = request.files['image']
    img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    cv2.imwrite('test.png',img)

    # Process landmark detection.
    res = detector.landmark_part_matching(img)

    # Return if no face detected.
    if len(res) == 0:
      return {'status':'fail','message':'No face detected'}    
      #return {'status': 'success', 'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]) , 'data': '[{\"id\":0,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":1,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":2,\"num\":0,\"rotation\":10,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":3,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":4,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":5,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":6,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0}]'}
    
    # Return status
    return {'status': 'success', 'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0]) , 'data': '[{\"id\":0,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":1,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":2,\"num\":0,\"rotation\":10,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":3,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":4,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":5,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0},{\"id\":6,\"num\":0,\"rotation\":0,\"faceScaleX\":1.0,\"faceScaleY\":1.0,\"facetransX\":0,\"facetransY\":0}]'}



@app.route('/finalize', methods=['POST'])
def finalize():
    print(request)



if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=19999)
    except KeyboardInterrupt:
        pass
    
    exit()
