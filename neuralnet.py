#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
#import matplotlib.pyplot as plt
import math as m


# In[ ]:


train_input = sys.argv[1]
validation_input = sys.argv[2]
train_out = sys.argv[3]
validation_out = sys.argv[4]
metrics_out = sys.argv[5]
num_epoch = int(sys.argv[6])
hidden_units = int(sys.argv[7])
init_flag = int(sys.argv[8])
learning_rate = float(sys.argv[9])


# In[2]:


# train_input='C:\\Users\\Anamika Shekhar\\data\\small_train.csv'
# validation_input='C:\\Users\\Anamika Shekhar\\data\\small_val.csv'
# train_out='C:\\Users\\Anamika Shekhar\\data\\tinyOut.csv'
# validation_out='C:\\Users\\Anamika Shekhar\\data\\tinyValidationOut.csv'
# metrics_out ='C:\\Users\\Anamika Shekhar\\data\\tinyMetric.txt'
# num_epoch=2
# hidden_units=4
# init_flag=2
# learning_rate=0.1


# In[3]:


def DataExtract(file_path):
    train_data,train_label = [],[]
    with open(file_path) as csv_file:
        line = csv_file.readlines()
        for i in line:
            i=i.strip()
            x=i.split(',')
            x.insert(1,1)
            train_data.append(list(map(float,x[1:])))
            train_label.append(int(float(x[0])))
    return train_data,train_label


# In[4]:


train_data,train_label=DataExtract(train_input)
validation_data,validation_label=DataExtract(validation_input)


# In[5]:


#np.shape(train_data)[1]-1 #(2,6)----6


# In[6]:


M=np.shape(train_data)[1]-1
K=4
if init_flag == 1:
    alpha = np.random.uniform(-0.1,0.1,(hidden_units, M+1))  
    beta = np.random.uniform(-0.1,0.1,(K,hidden_units+1)) 
    alpha[:,0] = 0.0
    beta[:,0] = 0.0
elif init_flag == 2:
    alpha = np.zeros((hidden_units, M+1))
    beta = np.zeros((K,hidden_units+1)) 


# In[ ]:





# In[ ]:





# In[7]:


def LinearForward(row_data,weight):
#     print("This is alpha",weight)
#     print("Shape of alpja",np.shape(weight))
#     print("Shape of row_data",np.shape(row_data))
#     print("This is row Data",row_data)
    return np.matmul(weight,row_data)


# In[8]:


def SigmoidForward(a):
    temp=[]
    z=[]
    z.append(1)
    #print("this is a",a)
    for i in range(0,len(a)):
        #print("This is i",i)
        #print(a[i])
        act=1/(1+np.exp(-a[i]))
        #print(act)
        z.append(act)
    return z
    


# In[9]:


def SoftmaxForward(b):
    softMax=[]
    N = [np.exp(i) for i in b]
    D = np.sum(N)
    for i in N:
        SM=i/D
        softMax.append(SM)
    return np.array(softMax)


# In[10]:


def CrossEntropyForward(y, yhat):
    l= 0.0
    for i in range(len(y)):
        l -= y[i] * np.log(yhat[i])
    return l


# In[11]:


def CrossEntropyBackward(y, yhat):
    gb = np.array(yhat - y)
    return gb


# In[12]:


def NNForward(x, y, alpha, beta):
    a = LinearForward(x,alpha)
    #print("Value of a (before sigmoid)",a)
    z = SigmoidForward(a)
    #print("this is z",z)
    b = LinearForward(z,beta)
    #print("Value of b (before softmax):",b)
    yhat = SoftmaxForward(b)
    #print("Value of y_hat (after softmax):",yhat)
    J = CrossEntropyForward(y, yhat)  
    #print("Cross entropy:",J)
#     print("Shape of a",np.shape(a))
#     print(a)
#     a=np.array(a).resize(len(a), 1)
    
#     print("Shape of a",np.shape(a))
#     z=np.array(z).resize(len(z), 1)
#     b=np.array(b).resize(len(b), 1)
#     ywhat=np.array(yhat).resize(len(yhat), 1)
#     x=np.array(x).resize(len(x), 1)
    return x, a, b, z, yhat, J
    


# In[13]:


def softMaxBackward(gb,z):
    gbeta=[]
    for i in range(0,len(gb)):
        x=list(gb[i]*np.array(z))
        gbeta.append(x)
    return np.array(gbeta)


# In[14]:


def NNBackward(x, y, alpha, beta,yhat,z):
    gb = CrossEntropyBackward(y, yhat)
    z=np.array(z)
    #print("this is z for NNBack",z)
    #print("d(loss)/d(b):",gb)
    gbeta= softMaxBackward(gb,z)
    #print("d(loss)/d(beta):",gbeta)
    betahat = np.delete(beta.T, 0, axis = 0)
    gz = betahat@gb
    #print("d(loss)/d(z):",gz)
    dz = z * (1-z)
    dz = np.delete(dz, 0,axis=0)
    ga = gz*dz
    #print("d(loss)/d(a):",ga)
    galpha = softMaxBackward(ga,x)
    #print("d(loss)/d(alpha):",galpha)
    return galpha, gbeta


# In[ ]:





# In[15]:


def lossplot(data,label,alpha, beta):
    s=0
    ps=0
    for i in range(len(data)):
        x1 = np.array(data[i])
        y1 = np.zeros(4)  #K
        #print(train_label[i])
        y1[label[i]] = 1
        x, a, b, z, yhat, J = NNForward(x1, y1, alpha, beta)
        s += J
        ps += J
    return s,ps


# In[16]:


def SGD(train_data,train_label,test_data,test_label,alpha, beta ,learning_rate,num_epoch):
    nalpha=np.zeros((hidden_units, M+1))
    nbeta=np.zeros((K,hidden_units+1)) 
    train_sum,train_loss_plot,test_sum,test_loss_plot = 0, 0,0,0
    tr_s,tr_l_p,te_s, te_l_p=0,0,0,0
    len_train, len_test = 0,0
    Ce_v=[]
    for e in range(1, num_epoch+1):
        for i in range(len(train_data)):
            x = np.array(train_data[i])
            y = np.zeros(4)  #K
            #print(train_label[i])
            y[train_label[i]] = 1
            x, a, b, z, yhat, J = NNForward(x, y, alpha, beta)
            #print(a)
            galpha, gbeta = NNBackward(x,y,alpha, beta, yhat, z)
            na=np.power(galpha, 2)
            nb=np.power(gbeta, 2)
            #print("this is nb",nb)
            nalpha+=na
            nbeta+=nb
            #print("this is nbeta:",nbeta)
            alphadash=nalpha+0.00001
            alphadash=(1/learning_rate)*np.sqrt(alphadash)
            betadash=nbeta+0.00001
            betadash=(1/learning_rate)*np.sqrt(betadash)
            #print("This is alphadash:",alphadash)
            alpha = alpha - np.divide(galpha,alphadash)
            #print("This is betadash",betadash)
            beta = beta - np.divide(gbeta,betadash)
#             print("Update alpha:",alpha)
#             print("Update beta:",beta)
        train_sum,train_loss_plot=lossplot(train_data,train_label,alpha, beta)
        tr_s=train_sum
        tr_l_p+=train_loss_plot
        test_sum,test_loss_plot=lossplot(test_data,test_label,alpha, beta)
        te_s=test_sum
        te_l_p+=test_loss_plot
#         print("train_sum,train_loss_plot:",train_sum,train_loss_plot)
#         print("test_sum,test_loss_plot:",test_sum,test_loss_plot)
        train_entropy = train_sum/len(train_data) 
        test_entropy = test_sum/len(test_data)
        len_train += len(train_data) 
        len_test += len(test_data)
        Ce_v.append(test_entropy)

        with open(metrics_out,"a") as f:
            f.write("epoch=" +str(e)+" crossentropy(train): " +str(train_entropy)+ "\n" + "epoch=" +str(e)                    + " crossentropy(test): " +str(test_entropy)+ "\n" ) 
    ce_train = tr_l_p/len_train
    ce_test = te_l_p/len_test
#     print("Error",ce_train,ce_test)
    return alpha,beta,ce_train,ce_test,Ce_v


# In[17]:


def Predict(train_data, train_label, alpha, beta,train_out):
    output_label = []
    output = ""
    for i in range(len(train_data)):
        x = np.array(train_data[i])
        y = np.zeros(4)
        y[train_label[i]] = 1
        x, a, b, z, yhat, J = NNForward(x, y, alpha, beta)
        predict=list(yhat)
        predict = predict.index(max(predict))
        output_label.append(predict)
    with open(train_out,"w") as f_out:
        for i in range(len(output_label)):
            f_out.write(str(output_label[i]))
            f_out.write('\n')
    return output_label


# In[18]:


with open(metrics_out,"w") as f:
       f.truncate()


# In[19]:


alpha,beta,CE_Train,CE_Test,Ce_v=SGD(train_data,train_label,validation_data,validation_label,alpha, beta ,learning_rate,num_epoch)
train_predict=Predict(train_data, train_label, alpha, beta,train_out)
validation_predict=Predict(validation_data,validation_label, alpha, beta,validation_out)


# In[20]:


#Ce_v


# In[21]:


error_train,error_test=0,0
for i in range(len(train_predict)):
    if train_predict[i] != train_label[i]:
        error_train += 1
for i in range(len(validation_predict)):
    if validation_predict[i] != validation_label[i]:
        error_test += 1
with open(metrics_out,"a") as f_out:
    f_out.write("error(train): " + str(error_train/len(train_label)) + "\n" + "error(test): " + str(error_test/len(validation_label)))


# In[22]:


#error_train/len(train_label)


# In[23]:


#error_test/len(validation_label)


# In[ ]:




