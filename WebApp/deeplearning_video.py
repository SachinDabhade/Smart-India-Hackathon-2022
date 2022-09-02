# Importing Libraries
from ast import main
from email.mime import image
from threading import main_thread
import cv2
# import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pprint import pprint
import os
import re
# from pathlib import Path
# from collections import OrderedDict, namedtuple
import easyocr
import datetime

# Path
BASE_PATH = os.getcwd()
# PREDICT_PATH = os.path.join(BASE_PATH,'static/predict/')
# ROI_PATH = os.path.join(BASE_PATH,'static/roi/')
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')


# LOAD YOLO MODEL
cuda = True
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
w = "./static/models/best2.onnx"  # Path of Weights
session = ort.InferenceSession(w, providers=providers)   # Initializing Model
reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory


# Drawing Box
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


# Detecting number plate
def get_detections(img):

    names = ['License Plate']

    # colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

    # ori_images = [img.copy()]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    im.shape

    outname = [i.name for i in session.get_outputs()]
    outname

    inname = [i.name for i in session.get_inputs()]
    inname

    inp = {inname[0]:im}


    start = datetime.datetime.now()
    outputs = session.run(outname, inp)[0]
    end = datetime.datetime.now()
    print('Required Time: ', end - start)
    print('Output: ', outputs)
    
    if len(outputs) == 0:
        return img

    Text = []
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        # image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        # print('Box: ', box)
        try:
            x, y = ((box[3]-box[1])//5), ((box[2]-box[0])//5)
            ROI = img[box[1]-x: box[3]+x, box[0]-y: box[2]+y]
        except Exception as E:
            print('Number plate is slightly have cutted edge')
            ROI = img[box[1]: box[3], box[0]: box[2]]
        # text = Plate_Recognizer(ROI)
        text = extract_text(ROI)
        # Text.append(text)
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = names[cls_id]
        print('Name', name)
        # color = colors[name]
        # print('Color: ', color)
        cv2.rectangle(img,box[:2],box[2:],[0, 255, 0],3)
        # cv2.putText(img,str(score),(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=3)
        cv2.putText(img, text,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
    return img


# Extract Text Using Plate Recognizer
def Plate_Recognizer(ROI):

    import requests
    from pprint import pprint

    Recognized = 'danger'
    R_status = 'Unsuccessfully'
    
    regions = ['in'] # Change to your country

    cv2.imwrite('pic.jpg', ROI)
    with open('pic.jpg', 'rb') as fp:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=regions),  # Optional
            files=dict(upload=fp),
            # headers={'Authorization': 'Token ac2e63ddb14a5a2472d495c479c4edd5b7111ec3'}) # Himanshu API Token: salunkehimanshu2001@gmail.com 
            headers={'Authorization': 'Token 15afe65bf7cb3ec1b01df2d4aa7fd562516291b1'}) # Sachin API Token: sachindabhade1922@gmail.com Dabhade@12
            # Add numbers of API keys here also ID and passwords also
        A = response.json()

    pprint(A)

    if A['results'] == []:
        return []

    else:
        number_plate_standard = {A['results'][0]['candidates'][i]['plate']: A['results'][0]['candidates'][i]['score'] for i in range(len(A['results'][0]['candidates']))}
        number_plate = {key: value for key, value in number_plate_standard.items() if re.match('^\w{2}\d{2}\w{2}\d{4}$', key) or re.match('^\w{2}\d{2}\w{3}$', key)}
        # re.match('^\w{2}\d{2}\w{2}\d{4}$', R) Our main matching pattern

        if len(number_plate) == 0:
            number_plate = [max(number_plate_standard, key=number_plate_standard.get).upper()]
            Recognized = 'success'
            return number_plate[0]
        else:
            number_plate = [max(number_plate, key=number_plate.get).upper()]
            print('Text Match Successfully: ', number_plate)
            Recognized = 'success'
            return number_plate[0]


# Text Extraction and Pattern Detection
def extract_text(ROI):
    # results = reader.readtext(ROI, detail=0)
    # R = ''
    # for result in results[::-1]:
    #     result = result.strip()
    #     R += result.upper()
    # if re.match('^\w{2}\d{2}\w{2}\d{4}$', R) or re.match('^\w{2}\d{2}\w{3}$', R):
    #     print('Text Match Successfully: ', R)
    R = 'Sachin'
    print('Detected Text: ', R)
    return R
