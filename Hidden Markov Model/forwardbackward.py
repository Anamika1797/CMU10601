#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math as m
import sys
import csv
from datetime import datetime


# In[2]:

'''
index_to_tag="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\index_to_tag.txt"
index_to_word ="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\index_to_word.txt"
validation="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\validation.txt"
hmminit="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\hmminit.txt"
hmmtrans="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\hmmtrans.txt"
hmmemit="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\hmmemit.txt"
predict_path="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\predict.txt"
metric_path="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\metric.txt"
predicted_given="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_output\\predicted.txt"

'''
# In[ ]:


validation= sys.argv[1]
index_to_word =  sys.argv[2]
index_to_tag = sys.argv[3]
hmminit = sys.argv[4]
hmmemit = sys.argv[5]
hmmtrans = sys.argv[6]
predict_path = sys.argv[7]
metric_path=sys.argv[8]


# In[3]:


index_Tag={}
tag_Index={}
with open(index_to_tag) as f_out:
    lines = f_out.readlines()
    #print(lines)
    for line in range(0,len(lines)):
        key=lines[line].replace("\n", "")
        index_Tag[key]=line
        tag_Index[line]=key   


# In[4]:


index_Word={}
word_Index={}
with open(index_to_word) as f_out:
    lines = f_out.readlines()
    #print(lines)
    for line in range(0,len(lines)):
        key=lines[line].replace("\n", "")
        index_Word[key]=line
        word_Index[line]=key 


# In[5]:


pi=[]
with open(hmminit) as f_out:
    for i in f_out.readlines():
        #print(i.replace("\n", ""))
        pi.append(float(i))
pi=np.array(pi)


# In[6]:


b=[]
B=[]
with open(hmmtrans) as f_out:
    for i in f_out.readlines():
        #print(i)
        x=i.replace("\n","").split(" ")
        #print(x)
        #x=np.delete(x,-1,0)
        #print(x)
        b.append(x)
        
#print(b)
for i in b:
    J=[]
    for j in i:
        J.append(float(j))
    B.append(J)


# In[7]:


a=[]
A=[]
with open(hmmemit) as f_out:
    for i in f_out.readlines():
        x=i.replace("\n","").split(" ")
        #x=np.delete(x,-1,0)

        a.append(x)
for i in a:
    J=[]
    for j in i:
        J.append(float(j))
    A.append(J)


# In[8]:


#len(A)


# In[9]:


#B


# In[10]:


seq=[]
with open(validation) as f_out:
    s=[]
    for i in f_out.readlines():
        if i == "\n":
            #print("Next Sequence Line")
            seq.append(s)
            s=[]
        else: 
            s.append(i.replace("\n", "").replace("\t","<<"))
    seq.append(s)
            


# In[11]:


#output_given=[]
#with open(predicted_given) as f_out:
 #   s=[]
  #  for i in f_out.readlines():
   #     if i == "\n":
    #        #print("Next Sequence Line")
     #       output_given.append(s)
      #      s=[]
       # else: 
        #    s.append(i.replace("\n", "").replace("\t","<<"))
    #output_given.append(s)
            


# In[12]:


predict=[]
ll=[]
LL=[]
acu=0
for s in seq:
    words=[]
    actual=[]
    tag_count=len(tag_Index.keys())
    #print(s)
    for i in s:
        words.append(i.split("<<")[0])
        actual.append(i.split("<<")[1])
    #print(actual)
    word_count=len(words)
    alpha = np.zeros((word_count,tag_count))
    for j in range(0,tag_count):
        alpha[0][j]= np.log(pi[j]) + np.log(A[j][index_Word[words[0]]])
    px=alpha[-1].sum()
    for t in range(1,word_count):
        for j in range(0,tag_count):
            sva=[]
            #print(alpha)
            for k in range(0,tag_count):
                #print(alpha[t - 1][k],np.log(B[k][j]))
                sva.append(alpha[t - 1][k] + np.log(B[k][j]))
            #print("For the word",s)
            ma=np.max(sva)
            sva1=[i - ma for i in sva]
            alpha[t][j]=np.log(A[j][index_Word[words[t]]])+ma+np.log(np.exp(sva1).sum())
    alphaT=np.array(alpha[-1])
    lm=np.max(alphaT)
    alphaT1=[i - lm for i in alphaT]
    LL.append(lm+np.log(np.exp(alphaT1).sum()))
    #ll.append(np.log(np.exp(alpha[-1]).sum()))
    ma=np.max(sva)
    sva1=[i - ma for i in sva]
    alpha[t][j]=np.log(A[j][index_Word[words[t]]])+ma+np.log(np.exp(sva1).sum())

    beta = np.zeros((word_count,tag_count))
    for j in range(0,tag_count):
        beta[-1][j]=0
    #print(beta.shape)
    for t in range(word_count-2,-1,-1):
        #print(t)
        for j in range(0,tag_count):
            svb=[]
            for k in range(0,tag_count):
                svb.append(np.log(A[k][index_Word[words[t + 1]]])+ beta[t + 1][k] + np.log(B[j][k]))
            mb=np.max(svb)
            svb1=[i - mb for i in svb]
            beta[t][j]=mb+np.log(np.exp(svb1).sum())

    p_log = alpha+beta
    p=np.exp(p_log)
    b=np.argmax(p_log, axis = 1)
    tag_predict=[]
    for i in b:
        tag_predict.append(tag_Index[i])
    #print(len(words))
    #print(len(tag_predict))
    predict_s=[]
    for i,j in zip(words,tag_predict):     
        predict_s.append(i+'<<'+j)
    predict.append(predict_s)
    for i,j in zip(actual,tag_predict):
        if i == j:
            acu=acu+1.0
    
    
        
        
            
    
    


# In[13]:


with open(predict_path,"w") as f_out:
    for i in predict:
        for j in i:
            word,label=j.split("<<")
            f_out.write(word)
            f_out.write("\t")
            f_out.write(label)
            f_out.write("\n")
        f_out.write("\n")
        


# In[14]:


#for i in range(0,len(predict)):
 #   for j in range(0,len(predict[i])):
  #      if(predict[j] != output_given[1][j]):
            #print(i)
   #         break;


# In[ ]:





# In[15]:


avgll=np.sum(LL)/len(seq)


# In[16]:


tot=0
acu=0
for i in range(0,len(seq)):
    tot+=len(seq[i])
    for j in range(0,len(seq[i])):
        true_label=seq[i][j].split("<<")[1]
        predict_label=predict[i][j].split("<<")[1]
        if true_label == predict_label:
            acu+=1
Acuracy=acu/tot


# In[17]:


with open(metric_path, "w") as f_out:
    Avgl= "Average Log-Likelihood: "+ str(avgll)
    Acu="Accuracy: "+ str(Acuracy)
    f_out.write(Avgl)
    f_out.write("\n")
    f_out.write(Acu)

