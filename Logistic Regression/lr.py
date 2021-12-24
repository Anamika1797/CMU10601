#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math as m
import sys
import csv
from datetime import datetime


# In[2]:


# formatted_train_input = "C:\\Users\\Anamika Shekhar\\OutputAnamika\\model1_formatted_train.tsv"
# formatted_validation_input =  "C:\\Users\\Anamika Shekhar\\OutputAnamika\\model1_formatted_valid.tsv"
# formatted_test_input = "C:\\Users\\Anamika Shekhar\\OutputAnamika\\model1_formatted_test.tsv"
# train_out = "C:\\Users\\Anamika Shekhar/OutputAnamika/train.labels"
# test_out = "C:\\Users\\Anamika Shekhar/OutputAnamika/test.labels"
# metric_out = "C:\\Users\\Anamika Shekhar\OutputAnamika\metrics.txt"
# dic_input = "C:\\Users\\Anamika Shekhar\\handout\\dict.txt"
# num_epoch=500


# In[ ]:


formatted_train_input = sys.argv[1]
formatted_validation_input =  sys.argv[2]
formatted_test_input = sys.argv[3]
dic_input = sys.argv[4]
train_out = sys.argv[5]
test_out = sys.argv[6]
metric_out = sys.argv[7]
num_epoch=int(sys.argv[8])


# In[3]:


Vocab={}
with open(dic_input, 'r') as tsv_file:                                                                                          
    tsv_read = csv.reader(tsv_file, delimiter="\t")
    for row in tsv_read:
        key, value = row[0].split(" ")
        Vocab[key]=value[:]


# In[4]:


def DataExtract1(path):
    labels=[]
    review=[]
    
    #c=0
    with open(path, 'r') as tsv_file:                                                                                          
        tsv_read = csv.reader(tsv_file, delimiter="\t")
        for row in tsv_read:
            small_review=[]
            small_review.append(1)
            labels.append(int(row[0]))
            
            for i in row[1:]:
                #print(i)
                small_review.append(int(i))
            review.append(small_review)    
        return(review,labels)


# In[5]:


def DataExtract2(path):
    labels=[]
    review=[]
    
    #c=0
    with open(path, 'r') as tsv_file:                                                                                          
        tsv_read = csv.reader(tsv_file, delimiter="\t")
        for row in tsv_read:
            small_review=[]
            small_review.append(1)
            labels.append(float(row[0]))
            
            for i in row[1:]:
                #print(i)
                small_review.append(float(i))
            review.append(small_review)    
        return(review,labels)


# In[ ]:





# In[ ]:





# In[6]:


def sparse_dot_product(index, weights):
    product=0.0
    for i in index:
        #print(i)
        #print("Weight is")
        #print(weights[i])
        product = product + weights[i]
        #print("product")
        #print(product)
    return product


# In[7]:


def sigmoid(product):
    #print("Help me",np.exp(product))
    x=np.exp(product) / (1 + np.exp(product))
    
    #print(x)
    return x


# In[8]:


def WeightCalculation2(review, labels):
    lr=0.01
    weights = np.zeros(len(review[0]))
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time for for output model2 =", current_time)
   
    for epoch in range(num_epoch):
        
        for index, label in zip(review, labels):
            #print(index)
            product = np.dot(np.array(index),np.array(weights))
            
            #print("This is prod",product)
            #print(label)
            #print(review_identity_vector)
            #print((np.exp(dot_product) / (1 + np.exp(dot_product))))
            #print("This is for train sigma")
            f=float(len(review))
            #print("This is length of each review",f)
            #alpha=(lr/f)*(label -sigmoid(product))
            #print(alpha)
            
            weights += (lr/f)*(label -(np.exp(product) / (1 + np.exp(product))))* np.array(index)
            #print("This is Weight",weights)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time for for output =", current_time)
   
    return weights


# In[9]:


def WeightCalculation1(Vocab,review, labels):
    lr=0.01
    weights = np.zeros(len(Vocab)+1)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time for for output for model1 =", current_time)
    for epoch in range(num_epoch):
#         now = datetime.now()
#         current_time = now.strftime("%H:%M:%S")
#         print("Current Time for for output =", current_time)
   
        for index, label in zip(review, labels):
            #print(index)
            product = np.dot(np.array(index),np.array(weights))
            
            #print("This is prod",product)
            #print(label)
            #print(review_identity_vector)
            #print((np.exp(dot_product) / (1 + np.exp(dot_product))))
            #print("This is for train sigma")
            f=float(len(review))
            #print("This is length of each review",f)
            #alpha=(lr/f)*(label -sigmoid(product))
            #print(alpha)
            
            weights += (lr/f)*(label -(np.exp(product) / (1 + np.exp(product))))* np.array(index)
            #print("This is Weight",weights)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time for for output =", current_time)
    return weights


# In[ ]:





# In[ ]:





# In[10]:


def predict(review,labels,weights):
    output_labels=[]
    for index, label in zip(review, labels):
            product = np.dot(np.array(index),np.array(weights))
            #print("Predict Sigma")
            prob = sigmoid(product)
            #print(prob)
            if(prob==0.5):
                print("what to do")
            if prob >= 0.5:
                output_labels.append(1)
            else:
                output_labels.append(0)
    return output_labels


# In[ ]:





# In[11]:


def ErrorOfOutput(original_data,predicted_data):
    error_count=0
    for i in range(0,len(original_data)):
        if original_data[i] !=predicted_data[i]:
            #print("Hello")
            error_count=error_count+1
    error_result=error_count/(len(original_data))
    return error_result


# In[12]:


train_review,train_labels=DataExtract2(formatted_train_input)
test_review,test_labels=DataExtract2(formatted_test_input)
valid_review,valid_labels=DataExtract2(formatted_validation_input)
if train_review[1][2]==0.0 or train_review[1][2]==1.0:
    weights=WeightCalculation1(Vocab,train_review,train_labels)
else:
    weights=WeightCalculation2(train_review,train_labels)
train_predict=predict(train_review,train_labels,weights)
test_predict=predict(test_review,test_labels,weights)
train_error=ErrorOfOutput(train_labels,train_predict)
test_error=ErrorOfOutput(test_labels,test_predict)    
train_error='{0:.2g}'.format(train_error)
test_error='{0:.2g}'.format(test_error)


# In[13]:


with open(metric_out, 'w') as f_out:
    f_out.write("error(train): " + str(train_error))
    f_out.write('\n')
    f_out.write("error(test): " + str(test_error))


# In[6]:


with open(test_out, 'w') as f_out:
    for i in test_predict:
        f_out.write(str(i))
        f_out.write('\n')


# In[7]:


with open(train_out, 'w') as f_out:
    for i in train_predict:
        f_out.write(str(i))
        f_out.write('\n')


# In[ ]:





# In[16]:





# In[17]:


# 

