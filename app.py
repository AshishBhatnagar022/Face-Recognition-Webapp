from __future__ import division,print_function
import sys
import os
import flask
import glob
import re
import numpy as np 
from flask import Flask,redirect,url_for,render_template
from werkzeug.utils import secure_filename
import cv2
import imutils
import os
from imutils import paths
import pickle
from vggFace import get_model
from keras.models import Model

app=Flask(__name__)
# app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0




@app.route('/',methods=['GET','POST'])
def index():
    uploaded_files = flask.request.files.getlist("file[]")
    length =len(uploaded_files)
    basepath = os.path.dirname(__file__)
    images=[]
    dispImg=[]
    img=""
    count=0
    for file in uploaded_files:
        file_path = os.path.join(
        basepath, 'static', secure_filename(file.filename))
        file.save(file_path)

        images.append(file_path)
    count=len(images)

    if not images:
    	images.append("static/images.png")   
    	images.append("static/images.png")
    else:
    	pass

    if count==0:
    	message='Please Select two images'
    elif count==1:
    	images=[]
    	message='Please Select two images'
    	images.append("static/images.png")   
    	images.append("static/images.png")
    else:
    	# dispImg=[]
    	print('Loading face recognizer...')
    	model=get_model()
    	model.load_weights('model/vgg_face_weights.h5')
    	faceModel = Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
    	print('DONE.....')

    	# for image in images:
    	img1,img2,message=getImages(images,faceModel)
    	images[0]=img1
    	images[1]=img2
    	
    return render_template('index.html',img=images,message=message)
def getImages(images,faceModel):
	finalImg=[]
	embeddings=[]
	dict={}
	epsilon=0.30
	img_size=224

	protoPath=os.path.sep.join(['model','deploy.prototxt'])
	modelPath=os.path.sep.join(['model','res10_300x300_ssd_iter_140000.caffemodel'])


	detector=cv2.dnn.readNetFromCaffe(protoPath,modelPath)
	
	a=0
	boxes=[]
	confidenceList=[]
	for countImg,image in enumerate(images):
		
		image = cv2.imread(image)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]
		imageBlob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
		detector.setInput(imageBlob)
		detections=detector.forward()
		count=0
		for i in range(0, detections.shape[2]):


		
			confidence= detections[0,0,i,2]


			if confidence>0.9:

				count+=1


				box=detections[0,0,i,3:7]*np.array([w,h,w,h])

				boxes.append(box)

				confidenceList.append(confidence)
				
		if not confidenceList:
			finalImg.append("static/images.png")
			message='Face not detected'
			continue
		else:
			
			maxi=np.argmax(confidenceList)
			(startX,startY,endX,endY)=boxes[maxi].astype('int')
			face=image[startY:endY,startX:endX]
			(fH,fW)=face.shape[:2]
			new_array=cv2.resize(face,(img_size,img_size))
		  
			we=new_array.reshape(-1,img_size,img_size,3)

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue
			embedding=faceModel.predict(we)[0,:]
			embeddings.append(embedding)

			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 255, 0), 2)
			
			cv2.imwrite('static/'+str(countImg)+'.jpg',image)
			image1='static/'+str(countImg)+'.jpg'
			finalImg.append(image1)
		if len(embeddings)>1:
			cosine_similarity = findCosineDistance(embeddings[0],embeddings[1])

			if(cosine_similarity < epsilon):
				message="They are same persons"
				print('cosine{} name {}'.format(cosine_similarity,"They are same"))
			else:
				message="They are different persons"
				print('cosine{} name {}'.format(cosine_similarity,"They are different"))

	return finalImg[0],finalImg[1],message
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

					
					
			

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__=='__main__':
	app.run(debug=False,threaded=False)