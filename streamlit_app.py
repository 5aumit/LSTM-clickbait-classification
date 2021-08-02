#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import texthero as hero


# In[9]:


def predict(message):
    model=load_model("saves\\tf_lstmmodel.h5")
    lemm = hero.clean(pd.Series([message])).tolist()
    with open('saves/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    x_1 = tokenizer.texts_to_sequences([lemm])
    x_1 = pad_sequences(x_1, maxlen=500)
    predictions = model.predict(x_1)[0][0]
    return predictions


# In[12]:


st.title('Clickbait Predictor')
message = st.text_area('Enter a title',"Type Here ..")
if st.button("Analyze"):
    with st.spinner("Analyzing"):
        prediction = predict(message)
    if prediction >= 0.5:
        st.warning("This title has a {:.2f}% chance of being clickbait".format(prediction))
    elif prediction <0.5:
        st.success("This title has a {:.2f}% chance of being clickbait".format(prediction))    
    


# In[ ]:




