#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd 
import numpy as np
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as ex
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud


# In[2]:


df_rahul=pd.read_csv("C:/Users/Ravi/Downloads/New projects/india polytical tweets/rahul_reviews.csv/rahul_reviews.csv")


# In[3]:


df_modi=pd.read_csv("C:/Users/Ravi/Downloads/New projects/india polytical tweets/modi_reviews.csv/modi_reviews.csv")


# In[4]:


df_rahul.head()


# In[5]:


df_modi.head()


# In[6]:


df_rahul.info()


# In[7]:


df_modi.info()


# 

# In[8]:


# Convert to String
df_modi['Tweet']=df_modi['Tweet'].astype(str)


# In[9]:


df_rahul['Tweet']=df_modi['Tweet'].astype(str)


# In[10]:


##Define Polarity Function: Defines a function that takes a text input (review) and returns its polarity (a measure of sentiment) using TextBlob.


# In[11]:


#Using TextBlob for finding Polarity of the Tweets
def find_polarity(review):
    return TextBlob(review).sentiment.polarity


# In[12]:


df_modi['Polarity']=df_modi['Tweet'].apply(find_polarity)


# In[13]:


df_modi


# In[14]:


df_rahul['Polarity']=df_rahul['Tweet'].apply(find_polarity)


# In[15]:


df_rahul


# In[16]:


# Label tweets based on polarity


# In[17]:


df_modi['label']=np.where(df_modi['Polarity']>0,'positive','negative')


# In[18]:


df_modi.loc[df_modi['Polarity']==0, 'label']='neutral'


# In[19]:


df_rahul['label']=np.where(df_rahul['Polarity']>0,'positive','negative')


# In[20]:


df_rahul.loc[df_rahul['Polarity']==0,'label']='neutral'


# In[21]:


df_modi


# In[22]:


df_rahul


# In[23]:


# Drop neutral tweets
df_modi=df_modi.drop(df_modi[df_modi['label']=='neutral'].index)


# In[24]:


df_rahul=df_rahul.drop(df_rahul[df_rahul['label']=='neutral'].index)


# In[25]:


df_modi.shape,df_rahul.shape


# In[26]:


len(df_modi[df_modi['label']=='positive'])


# In[27]:


len(df_rahul[df_rahul['label']=='positive'])


# In[28]:


#Making the Shape of Modi's and Rahul's dataframe the same
np.random.seed(10)
remove_np = 6041
remove_nn = 2440
drop_indicesp = np.random.choice((df_modi[df_modi['label'] == 'positive']).index,remove_np,replace = False)
drop_indicesn = np.random.choice((df_modi[df_modi['label'] == 'negative']).index,remove_nn,replace = False)

df_modi = df_modi.drop(drop_indicesp)
modi = df_modi.drop(drop_indicesn)


# In[29]:


np.random.seed(10)
remove_n = 367

drop_indices = np.random.choice(df_rahul.index,remove_n,replace = False)

rahul = df_rahul.drop(drop_indices)


# In[30]:


print(modi.shape)
print(rahul.shape)


# In[31]:


#Prediction of Positive and Negative Sentiments

modi.groupby('label').count()


# In[32]:


rahul.groupby('label').count()


# In[33]:


modi_count = modi.groupby('label').count()
neg_modi = (modi_count['Polarity'][0] / 1000) * 100
pos_modi = (modi_count['Polarity'][1] / 1000) * 100
rahul_count = rahul.groupby('label').count()
neg_rahul = (rahul_count['Polarity'][0] / 1000) * 100
pos_rahul = (rahul_count['Polarity'][1] / 1000) * 100
politicians = ['Modi','Rahul']

neg_list = [neg_modi,neg_rahul]
pos_list = [pos_modi,pos_rahul]


fig = go.Figure(
data = [
    go.Bar(name='Negative',x=politicians,y=neg_list),
    go.Bar(name='Positive',x=politicians,y=pos_list)
]
)
fig.update_layout(barmode='group')
fig.show()


# In[35]:


# Topic Modeling
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(modi['Tweet'])
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)


# In[37]:


# Word Cloud Visualization

import matplotlib.pyplot as plt
positive_tweets = " ".join(tweet for tweet in modi[modi['label'] == 'positive']['Tweet'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_tweets)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# In[40]:


# Assuming you have already preprocessed your data and have X (tweet text) and y (sentiment labels)
X = modi['Tweet']  # Tweet text
y = modi['label']  # Sentiment labels


# In[41]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[42]:


# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# In[43]:


# Initialize and train SVM classifier
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_vec, y_train)


# In[44]:


# Predict sentiment labels for test data
y_pred = svm_classifier.predict(X_test_vec)


# In[45]:


# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)


# In[ ]:




