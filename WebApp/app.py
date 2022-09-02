from flask import Flask, render_template, request
import os 
from deeplearning import object_detection

# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH,'static/upload/')

Detected = 'danger'
Recognized = 'danger'
D_status = 'Unsuccessfully'
R_status = 'Unsuccessfully'

@app.route('/',methods=['POST','GET'])
def index():
    if request.method == 'POST':
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        text_list, Detected, D_status, Recognized, R_status, New_method, details, Matched, Matched_Status = object_detection(path_save, filename)
        print('Last Detection: ', text_list, Detected, D_status, Recognized, R_status, New_method, details, Matched, Matched_Status)
        return render_template('index.html',upload=True,upload_image=filename,text=text_list,no=len(text_list), Detected=Detected, Recognized=Recognized, D_status=D_status, R_status=R_status, New_method=New_method, details=details, Matched=Matched, Matched_Status=Matched_Status)
         
    return render_template('index.html',upload=False)

if __name__ =="__main__":
    app.run(debug=True)