from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas
import numpy as np
#import sklearn
from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
import seaborn as sns

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""
#!/usr/bin/env python
# coding: utf-8

# In[10]:


#Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

s = pandas.read_csv("social_media_usage.csv")
s.shape


# In[3]:


#Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. 
#If it is, make the value of x = 1, otherwise make it 0. Return x. 

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


# In[4]:


#Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected
data = [['April', 10], ['Gus', 1], ['Brian', -1]]
toy = pandas.DataFrame(data, columns=['Name', 'test'])
clean_sm(toy["test"])


# In[5]:


#Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary 
#variable (that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) 
 #         which indicates whether or not the individual uses LinkedIn, and the following features: 
  #        income (ordered numeric from 1 to 9, above 9 considered missing), 
   #       education (ordered numeric from 1 to 8, above 8 considered missing), 
    #      parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). 
     #     Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.
s["marital"] = clean_sm(s["marital"])
s["gender"] = clean_sm(s["gender"])
s["par"] = clean_sm(s["par"])
s["sm_li"] = clean_sm(s["web1h"])

ss = s[(s["income"] < 10) &
       (s["educ2"] < 9) &
      (s["age"] < 99)]


# In[6]:


ss.head(5)


# In[51]:


#Create a target vector (y) and feature set (X)
x_ss = ss[["marital","age", "gender","par","educ2","income"]]
x_ss.shape
y_ss = ss['sm_li']
y_ss.shape


# In[52]:


#Split the data into training and test sets. Hold out 20% of the data for testing. 
#training_data = ss.sample(frac=0.8, random_state=25)
#testing_data = ss.drop(training_data.index)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_ss, y_ss, test_size=0.2, random_state=0)

#Explain what each new object contains and how it is used in machine learning


# In[53]:


#Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(class_weight = 'balanced')
#sklearn.LogisticRegression(class_weight=None).fit(training_data)


# In[54]:


logisticRegr.fit(x_train, y_train)


# In[55]:


logisticRegr.predict(x_test[0:10])


# In[56]:


predictions = logisticRegr.predict(x_test)


# In[57]:


score = logisticRegr.score(x_test, y_test)
print(score)


# In[58]:


#Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions 
#and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

cm = metrics.confusion_matrix(y_test, predictions)

plt.figure(figsize=(2,2))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);


# In[37]:


#Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each 
#quadrant represents


# In[43]:


#Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. 
#Use the results in the confusion matrix to calculate each of these metrics by hand. 
#Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. 
#After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[60]:


#Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), 
#with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? 
#How does the probability change if another person is 82 years old, but otherwise the same?
#["marital","age", "gender","par","educ2","income"]
logisticRegr.predict(np.array([[1,42,1,0,7,8]]))


# In[61]:


logisticRegr.predict(np.array([[1,82,1,0,7,8]]))


# In[48]:


import streamlit as st


# In[ ]:


educ = st.selectbox("Formal Education Level",
                   options= ["Less than high school",
                             "High school incomplete",
                             "High school graduate",
                             "Some college, no degree",
                             "Two-year associate degree",
                             "Four-year college",
                             "Some postgraduate",
                             "Postgraduate degree"])
st.write(f"Education: {educ}")

if educ == "Less than high school":
    educ2 = 1
elif educ == "High school incomplete":
    educ2 = 2
elif educ =="High school graduate":
    educ2 = 3
elif educ =="Some college, no degree":
    educ2 = 4
elif educ == "Two-year associate degree":
    educ2 = 5
elif educ =="Four-year college":
    educ2 = 6
elif educ =="Some postgraduate":
    educ2 = 7
elif educ =="Postgraduate degree":
    educ2 = 8
else: 
    educ2 = 0

income = st.selectbox("Select Income",
                   options= ["Less than $10,000",
                             "10 to under $20,000",
                             "20 to under $30,000",
                             "30 to under $40,000",
                             "40 to under $50,000",
                             "50 to under $75,000",
                             "75 to under $100,000",
                             "100 to under $150,000",
                             "$150,000 or more"])
st.write(f"Income:{income}")
if income == "Less than $10,000":
    incomenumber = 1
elif income == "10 to under $20,000":
    incomenumber = 2
elif income == "20 to under $30,000":
    incomenumber = 3
elif income == "30 to under $40,000":
    incomenumber = 4
elif income == "40 to under $50,000":
    incomenumber = 5
elif income == "50 to under $75,000":
    incomenumber = 6
elif income == "75 to under $100,000":
    incomenumber = 7
elif income == "100 to under $150,000":
    incomenumber = 8
elif income == "150,000 or more":
    incomenumber = 9
else: 
    incomenumber = 0

parent = st.selectbox("Are you a parent of a child under 18 living in your home?",
                   options= ["Yes",
                             "No"])
st.write(f"Parent:{parent}")
if parent == "Yes":
    par = 1
else: 
    par = 0

married = st.selectbox("Are currently married?",
                   options= ["Yes",
                             "No"])
st.write(f"Married:{married}")
if married == "Yes":
    marital = 1
else: 
    marital = 0

gender = st.selectbox("Gender",
                   options= ["Male",
                             "Female",
                            "Non-Binary",
                            "Other"])
st.write(f"Gender: {gender}")
if gender == "Male":
    gendernum = 1
else: 
    gendernum = 0

age = st.slider("Age",
                min_value=0,
                max_value=98,
                value=50)
st.write(f"Age: {age}")
#"marital","age", "gender","par","educ2","income"]

#marital = pandas.to_numeric(marital)
#age = pandas.to_numeric(age)
#gendernum = pandas.to_numeric(gendernum)
#par = pandas.to_numeric(par)
#educ2 = pandas.to_numeric(educ2)
#incomenumber = pandas.to_numeric(incomenumber)

list =np.array([[marital,age,gender,par,educ2,incomenumber]])
print(list)
logisticRegr.predict(list)

#if logisticRegr.predict(list) == 1:
 #   prediction = "You are a Linkedin user."
#else: prediction = "You are not a Linkedin user."

#st.write(f"Prediction: {prediction}")


# In[ ]:




