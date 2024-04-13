from flask import Flask, render_template, request,url_for,send_file
import numpy as np
import cv2
import os

app = Flask(__name__)

def colorize_image(image_path):

    image = cv2.imread(image_path)

    net = cv2.dnn.readNetFromCaffe('D:\Image Colorization Using Deep Learning\model\colorization_deploy_v2.prototxt', 'D:\Image Colorization Using Deep Learning\model\colorization_release_v2.caffemodel')
    pts = np.load('D:\Image Colorization Using Deep Learning\model\pts_in_hull.npy')

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    net.setInput(cv2.dnn.blobFromImage(L))

    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    return colorized

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():

    file = request.files['file']

    file_path = os.path.join('static', 'uploads', file.filename)
    file.save(file_path)

    colorized_image = colorize_image(file_path)

    colorized_filename = 'colorized_' + file.filename
    colorized_path = os.path.join('static', 'colorized', colorized_filename)
    cv2.imwrite(colorized_path, colorized_image)
    return render_template('result.html', image_path=url_for('static', filename='colorized/' + colorized_filename))

if __name__ == '__main__':
    app.run(debug=True)
