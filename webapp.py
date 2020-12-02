import streamlit as st
from streamlit.caching import cache
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
from PIL import Image

def main():
    gitam = Image.open("logo.png")
    st.sidebar.success("Navigation bar")
    status = st.sidebar.selectbox("GO TO",["Home","Detect Masked Face","UML Diagrams","Source Code"])
    st.sidebar.image(gitam,use_column_width=True)
    st.sidebar.markdown("# <font color='navy'>Created By:</font>",unsafe_allow_html=True)
    st.sidebar.markdown("### Sourav Mohanty",unsafe_allow_html=True)
    st.sidebar.info("Any contribution is appreciated")
    st.sidebar.markdown("[![](https://brandmark.io/logo-crunch/cache/40b2ff5a0672196bc4122a948ed00830_64_37a6259cc0c1dae299a7866489dff0bd.png)](https://github.com/souravcoder99) [![](https://brandmark.io/logo-crunch/cache/3e0c1e2b98ab847c77b9f84f4642b2c9_48_37a6259cc0c1dae299a7866489dff0bd.png)]()")
    
    if status == 'Home':
        st.title("Face Mask Detection Using Transfer Learning And OpenCV")
        st.markdown("---", unsafe_allow_html=True)
        picture = Image.open("SYL.png")
        st.image(picture,use_column_width=True)
    	

    elif status == 'Detect Masked Face':
        run_app()
    
    elif status == 'UML Diagrams':
        st.warning("hello")

    elif status == 'Source Code':
        st.code(get_file_content_as_string("webapp.py"))

def run_app():

    @st.cache
    def get_model():
        model = load_model("mask_recog.h5")
        return model

    model1 = get_model()
    resMap = {
            0 : 'Mask On ',
            1 : 'Mask Off '
        }
    colorMap = {
            0 : (0,255,0), 
            1 : (0,0,255)
        }
    st.title("Face-Mask-Detection App")
    st.write("<font color='navy'><b>This Web App lets you detect masked faces in an image</b></font>",unsafe_allow_html=True)
    file_image = st.file_uploader("Upload your Photos", type=['jpeg','jpg','png'])
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    if file_image is None:
        st.warning("You haven't uploaded any image file")
    else:
        file_bytes = np.asarray(bytearray(file_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.info("INPUT IMAGE")
        st.image(img, channels="BGR",use_column_width=True)
        faces = classifier.detectMultiScale(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),1.1,5,minSize=(60, 60))
        c=0
        t=len(faces)
        for face in faces:
            slicedImg = img[face[1]:face[1]+face[3],face[0]:face[0]+face[2]]
            slicedImg = cv2.cvtColor(slicedImg, cv2.COLOR_BGR2RGB)
            slicedImg = cv2.resize(slicedImg, (224, 224))
            slicedImg = img_to_array(slicedImg)
            slicedImg = np.expand_dims(slicedImg, axis=0)
            slicedImg =  preprocess_input(slicedImg)
            pred = model1.predict(slicedImg)
            acc = np.max(pred*100)
            s = str(acc)
            pred = np.argmax(pred)
            if pred == 0:
                c = c+1
            cv2.rectangle(img,(face[0],face[1]),(face[0]+face[2],face[1]+face[3]),colorMap[pred],2)
            cv2.putText(img, resMap[pred]+s+"%",(face[0],face[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        st.info("OUTPUT IMAGE")
        st.image(img, channels="BGR", use_column_width=True)
        st.success("Detection Accuracy : "+s+"%")
        st.info("{} people are wearing mask".format(c))
        st.error("{} people are not wearing mask".format(t-c))




if __name__ == "__main__":
    main()