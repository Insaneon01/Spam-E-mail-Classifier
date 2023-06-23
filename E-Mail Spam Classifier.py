#creating a user interface
#importing the streamlit an inbuild python library to host machine learning model to webpaage
import requests
import streamlit as st
from streamlit_lottie import st_lottie

#page title
st.set_page_config(page_title="Spam Email CLassifier WebPage",page_icon=":mirror_ball:",layout="wide")

#for animation
def load_lottieurl(url):
    r=requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding=load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

st.title("Spam E-Mail Classifier")

st.subheader("About Webpage")

#page content
left_column, right_column=st.columns(2)
with left_column:
    st.write("This is a webpage to find out the e-mail you receive is either spam e-mail or not")
    st.write("For to know a functioning is created you can access for free")
    st.write("You just have to enter the text received in the e-mail in the given box and click the predict button")
    st.write("Then it will tell either given email is spam or not")
with right_column:
    st_lottie(lottie_coding,height=300,key="coding")

st.subheader("The spam e-mail predictor")


#importing different libraries


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from collections import Counter

#importing the dataset

folder='C://Users//abhis//OneDrive//Desktop//Files//DataSets//emails//'
files=[]
files=os.listdir(folder)
emails=[folder + file for file in files]
words=[]

#cleansing the dataset
for email in emails:
    f=open(email,encoding='latin-1')
    blob=f.read()
    words+=blob.split(" ")

for i in range(len(words)):
    if not words[i].isalpha():
        words[i]=""

words_dict=Counter(words)

del words_dict[""]

words_dict=words_dict.most_common(3000)

features=[]
target=[]

#Thoroughly understanding the data and rifining for the better understanding for the machine learning model
for email in emails:
    f=open(email,encoding='latin-1')
    blob=f.read().split(" ")
    
    data=[]
    
    for i in words_dict:
        data.append(blob.count(i[0]))
    features.append(data)
    
    if 'spam' in email:
        target.append(0)
    if 'ham' in email:
        target.append(1)

#imput variable
X=np.array(features)
#output variable
y=np.array(target)

#spliting the data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)

#selecting and applying the model 
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

#training the model
classifier.fit(X_train,y_train)

#taking the input from the user
st.write('Predict Either e-mail is spam or not')

#input refinement
ask = st.text_input('Enter Email Text: ')
def convertor(ask):
    data=[]
    for i in words_dict:
        data.append(ask.split(" ").count(i[0]))
    emailInput=np.array(data)
    return emailInput.reshape(1,3000)
arr=convertor(ask)

#predicting the output
y_pred = classifier.predict(arr)

#using the streamlit library for the user interface for taking the inout and showing the result
ok = st.button('Predict')
if ok:
    if y_pred==0:
        st.write("Spam")   
    else:
        st.write("Not Spam")  







