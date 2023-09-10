from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cnn_predictor
import vgg_predictor
import resnet_predictor
import segment


import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


from tensorflow.keras.preprocessing.image import ImageDataGenerator
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # print('Original file name: ' + file.filename)
        # print('Sanitized file name: ' + filename)
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(file_path)
        # print('upload_image filename: ' + filename)

        print('Sanitized file name: ' + filename)
        name, ext = os.path.splitext(filename)
        print("File name without extension:", name)
        cnn_predicted_class = cnn_predictor.predict_class_cnn(file_path)
        # vgg_predicted_class = "Construction"
        print(cnn_predicted_class)
        vgg_predicted_class = vgg_predictor.predict_class_vgg(file_path)
        resnet_predicted_class = resnet_predictor.predict_class_resnet(file_path)

        image_id = name.lstrip("0")
        # predicted_path = segment.plot_image(name)
        predicted_path = segment.plot_image(int(image_id))
        print(predicted_path)
        print("cnn predicted class:", cnn_predicted_class)
        print("vgg predicted class:", vgg_predicted_class)
        print("resnet predicted class:", resnet_predicted_class)
        # flash('Image successfully uploaded and displayed below. Predicted classes: ' + vgg_predicted_class + ',' + resnet_predicted_class)
        return render_template('index.html', filename=filename, cnn_predicted_class=cnn_predicted_class, vgg_predicted_class=vgg_predicted_class, resnet_predicted_class=resnet_predicted_class,predicted_path=predicted_path)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)





 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 

@app.route('/predict/<filename>')
def predict_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    predicted_class = cnn_predictor.predict_class(file_path)
    return predicted_class

if __name__ == "__main__":
    app.run()