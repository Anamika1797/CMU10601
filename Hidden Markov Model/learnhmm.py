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
train = "C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\train.txt"
index_to_tag="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\index_to_tag.txt"
index_to_word ="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\index_to_word.txt"
validation="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\validation.txt"
hmminit="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\hmminit.txt"
hmmtrans="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\hmmtrans.txt"
hmmemit="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw7\\hw7\\handout\\fr_data\\hmmemit.txt"
'''

# In[ ]:


train = sys.argv[1]
index_to_word =  sys.argv[2]
index_to_tag = sys.argv[3]
hmminit = sys.argv[4]
hmmemit = sys.argv[5]
hmmtrans = sys.argv[6]


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


#seq[-1]


# In[ ]:





# In[6]:


seq=[]
with open(train) as f_out:
    s=[]
    for i in f_out.readlines():
        if i == "\n":
            #print("Next Sequence Line")
            seq.append(s)
            s=[]
        else: 
            s.append(i.replace("\n", "").replace("\t","<<"))
    seq.append(s)
            


# In[7]:


tag_count=int(len(index_Tag))
word_count=len(index_Word)
pi=np.ones(tag_count)
B=np.ones((tag_count,tag_count))
A=np.ones((tag_count,word_count))


# In[8]:


#seq


# In[9]:


x=[]
for s in seq:
    for i in range(0,len(s)):
        #print(s[i])
        yi,tag=s[i].split("<<")
        
        if i== 0:
            #print(tag)
            #print("Before",pi[index_Tag[tag]])
            pi[index_Tag[tag]] += 1
            x.append(tag)
            #print("After",pi[index_Tag[tag]])
        if i != len(s)-1:
            yi1,tag1=s[i+1].split("<<")
            B[index_Tag[tag]][index_Tag[tag1]] += 1
        A[index_Tag[tag]][index_Word[yi]] += 1


# In[10]:


pi /= np.sum(pi)
B /= np.sum(B, axis = 1).reshape(tag_count, -1)
A /= np.sum(A, axis = 1).reshape(tag_count, -1)


# In[11]:


#for i in p:
    
    #print(("%.18e"%(i)))


# In[12]:


#B


# In[ ]:





# In[ ]:





# In[13]:


with open(hmminit,"w") as f_out:
    for i in pi:
        f_out.write(str("%.18e"%(i)))
        f_out.write("\n")


# In[14]:


#for i in B:
#    print(i)
#    for j in i:
 #       print(j)


# In[15]:


with open(hmmtrans,"w") as f_out:
    for i in B:
        y=''
        for j in i:
            y+=str("%.18e"%(j))+" "
        #print(y)
        f_out.write(str.strip(str(y)))
        f_out.write("\n")
        


# In[16]:


with open(hmmemit,"w") as f_out:
    for i in A:
        z=''
        for j in i:
            z+=str("%.18e"%(j))+" "
        #print(z)
        f_out.write(str.strip(str(z)))
        f_out.write("\n")


# In[ ]:




