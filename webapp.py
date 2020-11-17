import streamlit as st
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np

model = load_model("mask_recog.h5")

resMap = {
        0 : 'Mask On ',
        1 : 'Mask Off '
    }

colorMap = {
        0 : (0,255,0),
        1 : (0,0,255)
    }

st.title("Face-Mask-Detection App")
st.write("This Web App lets you detect masked faces in an image")
file_image = st.sidebar.file_uploader("Upload your Photos", type=['jpeg','jpg','png'])

classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
if file_image is None:
    st.write("You haven't uploaded any image file")

else:
    file_bytes = np.asarray(bytearray(file_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR",use_column_width=True)
    faces = classifier.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(60, 60))
    for face in faces:
        
        slicedImg = img[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
        slicedImg = cv2.cvtColor(slicedImg, cv2.COLOR_BGR2RGB)
        slicedImg = cv2.resize(slicedImg, (224, 224))
        slicedImg = img_to_array(slicedImg)
        slicedImg = np.expand_dims(slicedImg, axis=0)
        slicedImg =  preprocess_input(slicedImg)
        pred = model.predict(slicedImg)
        acc = np.max(pred*100)
        s = str(acc)
        pred = np.argmax(pred)
        
        cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),colorMap[pred],2)
        cv2.putText(img, resMap[pred]+s+"%",(face[0],face[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    st.write("**Output Pencil Sketch**")
    st.image(img, channels="BGR", use_column_width=True)