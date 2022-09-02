from flask import Flask, render_template, request, Response, url_for
import os 
import cv2
from PIL import Image
from deeplearning_video import get_detections
from utilsis import RFID_MATCH

# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['video_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        # details, Matched, Matched_status = gen(path_save)
        # if Matched == 'success':
        #     return render_template('video_result.html', details=details, Matched_status=Matched_status)
        return Response(gen(path_save), mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template('index_video.html',upload=False)



def gen(path_save): 

    vid_capture = cv2.VideoCapture(path_save)
    
    if (vid_capture.isOpened() == False):
        print("Error opening the video file")

    # Read fps and frame count
    else:
        
        # Get frame rate information
        # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        fps = vid_capture.get(5)
        print('Frames per second : ', fps ,'FPS')

        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)

    while(vid_capture.isOpened()):

        # vid_capture.read() methods returns a tuple, first element is a bool and the second is frame
        ret, frame = vid_capture.read()

        if ret == True:
            
            print('\nImage Start')
            try:
                image, Text = get_detections(frame)
                # if len(Text) != 0:
                #     details, Matched, Matched_status = RFID_MATCH(Text)
                #     if Matched == 'success':
                #         return render_template('video_result.html', details=details, Matched_status=Matched_status)
                h, w, l = image.shape
                new_h = h//5
                new_w = w//5
                img = cv2.resize(image, (new_w, new_h)) 
                frame = cv2.imencode('.jpg', img)[1].tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + open('pic.jpg', 'rb').read() + b'\r\n')
            except Exception as E:
                print('Exception Occur: ', E)
            print('Image processing ends..\n')
        else:
            break

if __name__ =="__main__":
    app.run(debug=True)