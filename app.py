#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
st.set_page_config(page_title = 'Clickbait Detector')

import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import texthero as hero
import json
import numpy as np
import webbrowser


# In[9]:


model = model=load_model("saves\\tf_lstmmodel.h5")
def predict(message):
    t = hero.clean(pd.Series([message])).tolist()
    with open('saves\\tokenizer.json') as json_file:
        json_string = json.load(json_file)
    tokenizer = tokenizer_from_json(json_string)
    x_1  = np.array(tokenizer.texts_to_sequences(t))
    x_1 = pad_sequences(x_1, padding='post', maxlen=100)
    prediction = model.predict(x_1)[0][0]
    return prediction

# In[12]:


st.title('Clickbait Detector')
st.subheader('This app uses an LSTM Model to predict whether the given text is clickbait or not.')
message = st.text_area('Enter the title of an article or video from the internet.')
if st.button("Analyze"):
    with st.spinner("Analyzing"):
        prediction = predict(message)
    if prediction >= 0.5:
        st.warning("This title has a {:.2f}% chance of being clickbait".format(prediction*100))
    elif prediction <0.5:
        st.success("This title has a {:.2f}% chance of being clickbait".format(prediction*100))
        
url = 'https://github.com/5aumit/LSTM-clickbait-classification'
st.write("[GitHub](%s)"%url)
    


# In[ ]:




