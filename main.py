import streamlit as st
import numpy as np

from PIL import Image
import keras
from keras.utils import img_to_array,load_img
from keras.models import load_model
from streamlit_option_menu import option_menu

filenames_mk= ['M04_01.jpg',
               'M19_01.jpg',
               'M19_02.jpg',
               'M23_01.jpg',
               'M38_01.jpg',
               'M48_01.jpg',
               'M48_02.jpg']

filenames_others =['NM02_01.jpg',
                   'NM03_01.jpg',
                   'NM04_01.jpg',
                   'NM05_01.jpg',
                   'NM06_01.jpg',
                   'NM08_01.jpg',
                   'NM91_02.jpg']



with st.sidebar:
    choose = option_menu('App Gallery',['About','Monkeypox Images','Non Monkeypox Images','AI-Predict','Performance Metrics'],
                         icons=['house','image','image-fill','question-diamond-fill','speedometer'],
                         menu_icon='prescription2',default_index=0,
                         styles={
                             'container':{'padding':'5!important','background-color':'#fafafa'},
                             'icon':{'color':'orange','font-size':'25px'},
                             'nav-link':{'font-size':'16px','text-align':'left','margin':'0px','--hover-color':'#eee'},
                             'nav-link-selected':{'background-color':'#02ab21'}
                         })

if choose=='About':
    st.write("<h2> Monkeypox: Binary Classification Model</h2>",unsafe_allow_html=True)
    st.write("")
    st.write("The dataset is collected from Kaggle, it include **102 Monkeypox** images and **126 for others**. This is a binary classification problem to predict Monkeypox Vs Measles or Chickenpox and we will use Deep Learning in Computer Vision with Tensorflow and Keras to build the model architecture")

elif choose =='Monkeypox Images':
    st.write("<div align='center'><h3>Monkeypox Images</h3></div>",unsafe_allow_html=True)
    st.write("")
    col1,col2,col3=st.columns(3)
    with col1:
        for i in range(0,2):
            img = Image.open(filenames_mk[i])
            st.image(img)

    with col2:
        for i in range(2,4):
            img=Image.open(filenames_mk[i])
            st.image(img)

    with col3:
        for i in range(4,6):
            img3=Image.open(filenames_mk[i])
            st.image(img3)

elif choose=='Non Monkeypox Images':
    st.write("<div align='center'><h3> Non Monkeypox Images</h3></div>",unsafe_allow_html=True)
    st.write("<div align='center'><h4> Measles or Chickenpox </h4></div>",unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    with col1:
        for i in range(0,2):
            img=Image.open(filenames_others[i])
            st.image(img)

    with col2:
        for i in range(2,4):
            img=Image.open(filenames_others[i])
            st.image(img)

    with col3:
        for i in range(4,6):
            img=Image.open(filenames_others[i])
            st.image(img)




elif choose =='AI-Predict':
    model=load_model("m_pox1.h5")
    image1_path=[("monkeypox1.jpg",'Monkeypox')]
    image2_path=[("monkeypox2.jpg",'Monkeypox')]
    image3_path=[("others1.jpg",'Measles or Chickenpox')]
    image4_path =[("others2.jpg",'Measles or Chickenpox')]
    st.title('Image Classification')
    class_names=['Monkeypox','Others']
    uploaded_file=st.file_uploader('',type=['jpg','jpeg','png'])
    if st.button("Predict"):
        if uploaded_file is not None:
            img=load_img(uploaded_file,target_size=(224,224))
            img=img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img= img/255.0
            pred=model.predict(img)
            arg=np.argmax(pred)
            pred_int = pred[arg][0]
            if pred_int>0.5:
                st.write(f"The model is {round(pred_int*100,2)}% confident that the image has **No signs of Monkeypox**")
            else:
                st.write(f"The model is {round((1-pred_int)*100,2)}% confident that the image shows **signs of Monkeypox**")



    col1,col2,col3,col4=st.columns(4)
    with col1:
        for image,label in image1_path:
            img=Image.open(image)
            st.image(img,caption=label)

    with col2:
        for image,label in image2_path:
            img=Image.open(image)
            st.image(img,caption=label)


    with col3:
        for image,label in image3_path:
            img=Image.open(image)
            st.image(img,caption=label)
    with col4:
        for image,label in image4_path:
            img=Image.open(image)
            st.image(img,caption=label)


elif choose=='Performance Metrics':
    st.write("<div align='center'><h3>Performance Metrics</h3><div>",unsafe_allow_html=True)
    col1,col2,col3=st.columns(3)
    with col1:
        if st.button('Accuracy'):
            st.write("The Model overall accuracy is **75%**")
    with col2:
        if st.button('Precision'):
            st.write("The precision tells us the percentage of the model's positive predictions that are correct.")
            st.write("The model precision is **60%**")
    with col3:
        if st.button('Recall'):
            st.write("The recall  tells us the percentage of positive samples that were correctly identified by the model.")
            st.write(" The model recall is **88%**")






