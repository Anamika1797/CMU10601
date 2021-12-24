#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import environment as env
from environment import MountainCar as mc
import numpy as np


# In[2]:
'''

global mode
mode='tile'
global weight_out
weight_out="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw8\\handout\\weight_out.out"
global returns_out
returns_out="C:\\Users\\Anamika Shekhar\\Desktop\\Fall21\\10601\\week9\\hw8\\handout\\returns_out.out"

global episodes
episodes=20
global max_iteration
max_iteration=200
global epsilon
epsilon=0.05
global gamma
gamma=0.99
global learning_rate
learning_rate=0.00005
global a_list
a_list=[0,1,2]
'''

# In[ ]:


global mode
mode=sys.argv[1]
global weight_out
weight_out=sys.argv[2]
global returns_out
returns_out=sys.argv[3]

global episodes
episodes=int(sys.argv[4])
global max_iteration
max_iteration=int(sys.argv[5])
global epsilon
epsilon=float(sys.argv[6])
global gamma
gamma=float(sys.argv[7])
global learning_rate
learning_rate=float(sys.argv[8])
global a_list
a_list=[0,1,2]


# In[16]:


env = mc(mode);
#env.reset()


# In[4]:


global w
global b
b=0
if mode == "raw":
    w = np.zeros((3,2))
    
else:
    w = np.zeros((3,2048))


# In[5]:


def TransformData(cs):
    if mode == "raw":
        s=np.zeros(2)
        for i in cs.keys():
            s[i]=cs[i]
        s.reshape(1,2)
    else:
        s=np.zeros(2048)
        for i in cs.keys():
            s[i]=cs[i]
        s.reshape(1,2048)
    return s


# In[6]:


#w


# In[7]:


#q=predict(s)
#a=action(s,q)
#update(s,a,2)


# In[8]:


#x=np.array(curr_state.values())
#x


# In[ ]:





# In[ ]:





# In[9]:


def predict(s):
    q=[]
    for i in a_list:
        #print(w[i].shape)
        #print(np.transpose(s).shape)
        q.append(np.dot(w[i],np.transpose(s))+b)
    return q
#print("Anamika hit Sneha with a chappal")   
def action(s,q):
    random_choice = np.random.rand()
    if random_choice < epsilon:
        a=np.random.choice(3)
    else:
        a=np.argmax(q)
    return a
def update(s,a,temp_diff):
    global w,b
    #print("This is w",w[a])
    #print("this is s",s)
    #print(np.multiply(s,learning_rate*temp_diff))
    if(mode == 'tile'):
        for i in range(0,len(s)):
            w[a][i]=w[a][i]-s[i]*learning_rate*temp_diff
    else:
        w[a] = w[a] - np.multiply(s,learning_rate*temp_diff);
    #print(b)
    b = b - learning_rate*temp_diff
    #print(b)
    return w,b

    
    
    
    


# In[10]:


return_list=[]
for e in range(episodes):
    sum_reward=0
    list_reward=[]
    env.reset()
    for i in range(max_iteration):
        #print("This is iternation",i)
        si = env.transform(env.state)
        si=TransformData(si)
        #print("This is State:",si)
        #print("This is the weight:",w)
        qi=predict(si)
        #print("This is q:",qi)
        ai=action(si,qi)
        #print("This is action:",ai)
        si1,reward,done=env.step(ai)
        #print("This is the next State:",si1)
        si1=TransformData(si1)
        qi1=predict(si1)
        list_reward.append(reward)
        sum_reward += reward
        t=reward + gamma*np.max(qi1);
        td=qi[ai]-t
        #print("Temp Difference:",td)
        update(si,ai,td)
        if done:
            break;
    return_list.append(sum_reward)
    
        
        
        
    
    


# In[11]:


with open(weight_out, 'w') as f_out:
    f_out.write(str(b))
    f_out.write('\n')
    for i in np.transpose(w):
        for j in i:
            #print(j)
            f_out.write(str(j))
            f_out.write('\n')
#     f_out.write("error(test): " + str(test_error))


# In[12]:


#for i in np.transpose(w):
 #   for j in i:
  #      print(j)


# In[13]:


with open(returns_out, 'w') as f_out:
    for i in return_list:
        f_out.write(str(i))
        f_out.write('\n')


# In[14]:


#b

