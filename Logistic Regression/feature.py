#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math as m
import sys
import csv
from datetime import datetime


# In[2]:


# train_input = "C:\\Users\\Anamika Shekhar\\handout\\smalldata\\train_data.tsv"
# validation_input = "C:\\Users\\Anamika Shekhar\\handout\\smalldata\\valid_data.tsv"
# test_input = "C:\\Users\\Anamika Shekhar\\handout\\smalldata\\test_data.tsv"
# dic_input = "C:\\Users\\Anamika Shekhar\\handout\\dict.txt"
# formatted_train_out = "C:\\Users\\Anamika Shekhar\\OutputAnamika\\model1_formatted_train.tsv"
# formatted_validation_out =  "C:\\Users\\Anamika Shekhar\\OutputAnamika\\model1_formatted_valid.tsv"
# formatted_test_out = "C:\\Users\\Anamika Shekhar\\OutputAnamika\\model1_formatted_test.tsv"
# feature_flag = 2
# feature_dict_input="C:\\Users\\Anamika Shekhar\\handout\\word2vec.txt"

train_input = sys.argv[1]
validation_input = sys.argv[2]
test_input = sys.argv[3]
dic_input = sys.argv[4]
formatted_train_out = sys.argv[5]
formatted_validation_out = sys.argv[6]
formatted_test_out = sys.argv[7]
feature_flag = int(sys.argv[8])
feature_dict_input= sys.argv[9]


# In[3]:


Vocab={}
with open(dic_input, 'r') as tsv_file:                                                                                          
    tsv_read = csv.reader(tsv_file, delimiter="\t")
    for row in tsv_read:
        key, value = row[0].split(" ")
        Vocab[key]=value[:]


# In[4]:


def WordtoVector(feature_dict_input):
    Word2Vec={}
    with open(feature_dict_input, 'r') as tsv_file:                                                                                          
        tsv_read = csv.reader(tsv_file, delimiter="\t")
        for row in tsv_read:
            #key, value = row[0].split(" ")
            key=row[0]
            value=row[1:]

            Word2Vec[key]= np.array(value).astype(float)
    return Word2Vec


# In[ ]:





# In[ ]:





# In[5]:


def Data_Extract(file_path):
    labels=[]
    review=[]
    rr=[]
    with open(file_path, 'r') as tsv_file:                                                                                          
        tsv_read = csv.reader(tsv_file, delimiter="\t")
        for row in tsv_read:
            r = row[0].split(" ")
            ##rr.append(r)
            labels.append(r[0])
            review.append(row[1].split(" "))
    return review,labels


# In[6]:


#labels,review=Data_Extract(train_input)


# In[7]:


#labels


# In[8]:


#Fin_Feacture_Vec=Model1(Vocab,review)


# In[9]:


def Model1(review,labels,Vocab):
    VectorMain=[]
    for sen,label in zip(review,labels):
        VectorSen = np.zeros(len(Vocab))
        #print(sen)
        for word in sen:
            if word in Vocab.keys():
                i=int(Vocab[word])
                #print(i)
                VectorSen[i]=1
        VectorMain.append(list(VectorSen))
    return VectorMain        
     


# In[10]:


#len(VectorMain[0])


# In[11]:


def Model2(review,labels,Word2Vec):
    VectorMain=[]
    for sen,label in zip(review,labels):

        Count_Dic={}
        for word in sen:

            if word in Word2Vec.keys():
                if word not in Count_Dic.keys():
                    Count_Dic[word] = 1
                else:
                    Count_Dic[word] += 1
        S = np.zeros(300)
        J=0
        for i in Count_Dic.keys():
            #print(Count_Dic[i])
            x=str(i)
            #print(i)
            #print(np.array(Word2Vec[i]))
            i=int(Count_Dic[i])*np.array(Word2Vec[i])
            S= S+i
            #print(S[1])
            z=Count_Dic[x]
            J=J+z
        ResultArray=(1/J)*S
        VectorMain.append(list(ResultArray)) 
    return VectorMain
        
        
        
            


# In[12]:


def outputModel2(path,labels,VectorMain,Word2Vec):
    file = open(path, "w")
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    #print("Current Time for for output =", current_time)
    for label, feat in zip(labels, VectorMain):
        line = str(label)+'.000000'
        for value in feat:
                line += "\t"
                #print(round(value,6))
                line += str(round(value,6))
            #print(line)
        line += "\n"
        #print(line)
        file.write(line)
    file.close()


# In[13]:


def outputModel1(path,labels,VectorMain,Vocab):
    file = open(path, "w")
    for i in range(0,len(labels)):
        #print("This is index",i)
        #print("This is label",labels[i])
        line = str(int(labels[i]))
        for j in range(0,len(Vocab)):
            line += "\t"
            #print(j)
            line += str(int(VectorMain[i][j]))
        line += "\n"
        file.write(line)
    file.close()


# In[14]:


if feature_flag==1:
    train_review,train_labels=Data_Extract(train_input)
    valid_review,valid_labels=Data_Extract(validation_input)
    test_review,test_labels=Data_Extract(test_input)
    format_train=Model1(train_review,train_labels,Vocab)
    format_valid=Model1(valid_review,valid_labels,Vocab)
    format_test=Model1(test_review,test_labels,Vocab)
    outputModel1(formatted_train_out,train_labels,format_train,Vocab)
    outputModel1(formatted_validation_out,valid_labels,format_valid,Vocab)
    outputModel1(formatted_test_out,test_labels,format_test,Vocab)
elif feature_flag==2:
    train_review,train_labels=Data_Extract(train_input)
    valid_review,valid_labels=Data_Extract(validation_input)
    test_review,test_labels=Data_Extract(test_input)
    Word2Vec=WordtoVector(feature_dict_input)
    format_train=Model2(train_review,train_labels,Word2Vec)
    format_valid=Model2(valid_review,valid_labels,Word2Vec)
    format_test=Model2(test_review,test_labels,Word2Vec)
    outputModel2(formatted_train_out,train_labels,format_train,Word2Vec)
    outputModel2(formatted_validation_out,valid_labels,format_valid,Word2Vec)
    outputModel2(formatted_test_out,test_labels,format_test,Word2Vec)
 


# In[ ]:





# In[ ]:





# In[ ]:




