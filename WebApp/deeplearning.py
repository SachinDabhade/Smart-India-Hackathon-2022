# Importing Libraries
from cgitb import text
import cv2
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pprint import pprint
import os
import re
import easyocr
import datetime
from utilsis import RFID_MATCH

# Path
BASE_PATH = os.getcwd()
PREDICT_PATH = os.path.join(BASE_PATH,'static/predict/')
ROI_PATH = os.path.join(BASE_PATH,'static/roi/')
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')


# LOAD YOLO MODEL
cuda = False
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
    colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

    ori_images = [img.copy()]
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
    
    print('Output length: ', len(outputs))

    if len(outputs) == 0:
        return ori_images[0], outputs, []

    roi = []
    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        roi.append(box.copy())
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = names[cls_id]
        color = colors[name]
        cv2.rectangle(image,box[:2],box[2:],color,3)
        cv2.putText(image,str(score),(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  

    return ori_images[0], outputs, roi


# Extract Text Using Plate Recognizer
def Plate_Recognizer(img_path):

    import requests
    from pprint import pprint

    Recognized = 'danger'
    R_status = 'Unsuccessfully'
    
    regions = ['in'] # Change to your country

    with open(img_path, 'rb') as fp:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=regions),  # Optional
            files=dict(upload=fp),
            headers={'Authorization': 'Token ac2e63ddb14a5a2472d495c479c4edd5b7111ec3'})

        A = response.json()
    print('A:', A)
    if A['results'] == []:
        return A['results'], Recognized, R_status

    else:
        result = []
        number_plate_standard = {A['results'][0]['candidates'][i]['plate']: A['results'][0]['candidates'][i]['score'] for i in range(len(A['results'][0]['candidates']))}
        number_plate = {key: value for key, value in number_plate_standard.items() if re.match('^\w{2}\d{2}\w{2}\d{4}$', key) or re.match('^\w{2}\d{2}\w{3}$', key) or re.match('\w{2}\d{2}\w{1}\d{4}$' , key)}
        # re.match('^\w{2}\d{2}\w{2}\d{4}$', R) Our main matching pattern
        
        if len(number_plate) == 0:
            number_plate = [max(number_plate_standard, key=number_plate_standard.get).upper()]
            print('Max Probability Applied', number_plate)
            result.append(number_plate[0])
        else:
            number_plate = [max(number_plate, key=number_plate.get).upper()]
            print('Text Match Successfully: ', number_plate)
            result.append(number_plate[0])
            Recognized = 'success'
            R_status = 'Successfully'
    return result, Recognized, R_status


# Text Extraction and Pattern Detection
def extract_text(ROI):
    results = reader.readtext(ROI, detail=0)
    Recognized = 'danger'
    R_status = 'Unsuccessfully'
    R = ''
    for result in results[::-1]:
        result = result.strip()
        R += result.upper()
    if re.match('^\w{2}\d{2}\w{2}\d{4}$', R):
        print('Text Match Successfully: ', R)
        Recognized = 'success'
        R_status = 'Successfully'
    print('Detected Text: ', R)
    return [R], Recognized, R_status


# predictions
def yolo_predictions(img, filename):

    # variables
    Detected = 'danger'
    D_status = 'Unsuccessfully'
    Recognized = 'danger'
    R_status = 'Unsuccessfully'

    ## step-1: detections
    print('Entered in step 1')
    image, outputs, Roi = get_detections(img)
    cv2.imwrite('./static/predict/{}'.format(filename), image)
    print("Outputs: ", outputs)
    if (len(Roi) == 0):
        New_method = 'No number plate detected... Trying different method...'
        path = os.path.join(UPLOAD_PATH, filename)
        text, Recognized, R_status = Plate_Recognizer(path)
        print('Testing1: ', text, Detected, D_status, Recognized, R_status, New_method)
        return text, Detected, D_status, Recognized, R_status, New_method

    ## step-2: Finding Region of Interest
    print(Roi)
    Detected = 'success'
    D_status = 'Successfully'
    print('Entered in step 2')
    Text = []
    for i, roi in enumerate(Roi):
        try:
            x, y = ((roi[3]-roi[1])//5), ((roi[2]-roi[0])//5)
            ROI = img[roi[1]-x: roi[3]+x, roi[0]-y: roi[2]+y]
            print('ROI: ', len(ROI))
        except Exception as E:
            print('Number plate is slightly have cutted edge')
            ROI = img[roi[1]: roi[3], roi[0]: roi[2]]

        ## step-3: Text Extraction and Pattern Matching
        print('Entered in step 3')
        new_filename = filename.split('.')[0]+'.jpg'
        cv2.imwrite('./static/roi/{}'.format(str(i)+new_filename), ROI)
        path = os.path.join(ROI_PATH, str(i)+new_filename)
        text, Recognized, R_status = Plate_Recognizer(path)
        # text, Recognized, R_status = extract_text(ROI)
        try:
            Text.append(text[0])
        except Exception as E:
            print('No Text Found... Sorry..')
    
    print('Testing2: ', text, Detected, D_status, Recognized, R_status)
    return Text, Detected, D_status, Recognized, R_status, None


def object_detection(path, filename):
    # read image
    image = cv2.imread(path) # PIL object
    image = np.array(image, dtype=np.uint8) # 8 bit array (0,255)
    text_list, Detected, D_status, Recognized, R_status, New_method = yolo_predictions(image, filename)
    details, Matched, Matched_Status = RFID_MATCH(text_list)
    if (New_method == None):
        return text_list, Detected, D_status, Recognized, R_status, '...', details, Matched, Matched_Status
    return text_list, Detected, D_status, Recognized, R_status, New_method, details, Matched, Matched_Status


def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf                                  

def RFID_Match(number_plate):
    pass