#!/usr/bin/env python
# coding: utf-8

# In[20]:


#!/usr/bin/env python
# coding: utf-8

# In[8]:


#!/usr/bin/env python
# coding: utf-8

# In[510]:


import numpy as np
import csv
import math
import sys


# In[511]:


class Node(object):
    def __init__(self,node):
        self.node="null"
        self.Countclass1=0
        self.Countclass2=0
        self.classification="null"
        self.LeftChild= None
        self.RightChild= None
        self.NodesData=[]
        self.isLeaf='N'

    def MajorityVote(self):
        #print(self.Countclass1,self.Countclass2)
        if self.Countclass1>=self.Countclass2:
            self.classification=labels[0]
        elif self.Countclass1<self.Countclass2:
            self.classification=labels[1]
        #print(self.classification)

if __name__ == "__main__":     
    
    data = []
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_Depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metric_out = sys.argv[6]
    print("The input file is: %s" % (train_input))
    print("The input file is: %s" % (test_input))
    print("The input file is: %s" % (max_Depth))
    print("The output file is: %s" % (train_out))
    print("The output file is: %s" % (test_out))
    print("The output file is: %s" % (metric_out))

 




    
    with open(train_input, 'r') as tsv_file:                                                                                          
        tsv_read = csv.reader(tsv_file, delimiter="\t")
        for row in tsv_read:
            data.append(row)


    # In[520]:


    train_data=data
    features=data[0][:-1]
    features_left=data[0][:-1]
    labels=list(np.unique(np.asarray(data).transpose()[-1][1:]))
    leafValues=list(np.unique(np.asarray(data).transpose()[0][1:]))
    Totalfeatures=len(features)
    Left_Child = [];
    Right_Child = [];
    features_left


    # In[440]:


    

    # In[514]:


    def Assign_Key_Count(Key):
        Key.Countclass1=0
        Key.Countclass2=0
        ##print("HEY!! this is in calc count")
        ##print(Key.NodesData)
        ##print(labels)
        for i in Key.NodesData:
            ##print(i[-1])
            if i[-1]==labels[0]:
                Key.Countclass1=Key.Countclass1+1
            elif i[-1]==labels[1]:
                Key.Countclass2=Key.Countclass2+1


    # In[515]:


    def CountLabel(Training_data_l):
        global train_data
        Countclass1=0
        Countclass2=0
        #print("Inside the count label")
        #print(Training_data_l)
        for i in (Training_data_l):
            if i[-1]==labels[0]:
                Countclass1=Countclass1+1
            elif i[-1]==labels[1]:
                Countclass2=Countclass2+1
        #print(Countclass1, Countclass2)


        return [Countclass1,Countclass2]


    # In[516]:




    # In[402]:


    def MutualInfo(Training_data_l,Attribute):
        #print(Attribute)
        for i in range(0,len(train_data[0])):
            #print(i)
            if (str(Attribute) == str(train_data[0][i])):   
                break;
        index = i
        ##print(index, " this is the index for feature", Attribute)
        #print(index)
        num_features = len(train_data[0])
        count_leaf1=0.0
        count_leaf2=0.0
        count_leaf1_label0=0.0
        count_leaf1_label1=0.0
        count_leaf2_label0=0.0
        count_leaf2_label1=0.0
        count_leaf2_label2=0.0
        for row in Training_data_l:
            if(row[index] == leafValues[0]):
                count_leaf1 +=1;
                if(row[-1] == labels[0]):
                    count_leaf1_label0 += 1
                if(row[-1] == labels[1]):
                    count_leaf1_label1 += 1
            if(row[index] == leafValues[1]):
                count_leaf2 +=1;
                if(row[-1] == labels[0]):
                    count_leaf2_label0 += 1
                if(row[-1] == labels[1]):
                    count_leaf2_label1 += 1
        #print(count_leaf1,count_leaf2,count_leaf1_label0,count_leaf1_label1,count_leaf2_label0,count_leaf2_label1,
        #print(count_leaf1,count_leaf2,count_leaf1_label0,count_leaf1_label1,count_leaf2_label0,count_leaf2_label1,
              ##count_leaf2_label2)

        ##print(count_leaf1, count_leaf2, count_leaf1+ count_leaf2, count_leaf1_label0, count_leaf1_label1, count_leaf2_label0, count_leaf2_label1)

        P_leaf1 = 0;
        P_leaf2 = 0;
        P_leaf1_label0 = 0;
        P_leaf1_label1 = 0;
        P_leaf2_label0 = 0;
        P_leaf2_label1 = 0;

        if((count_leaf1 + count_leaf2) != 0):
            P_leaf1 =count_leaf1/(count_leaf1+count_leaf2)
            P_leaf2 =count_leaf2/(count_leaf1+count_leaf2)
        if(P_leaf1):
            P_leaf1_label0=count_leaf1_label0/(count_leaf1_label0+count_leaf1_label1)
            P_leaf1_label1=count_leaf1_label1/(count_leaf1_label0+count_leaf1_label1)
        if(P_leaf2):
            P_leaf2_label0=count_leaf2_label0/(count_leaf2_label0+count_leaf2_label1)
            P_leaf2_label1=count_leaf2_label1/(count_leaf2_label0+count_leaf2_label1)

        ##print(P_leaf1, P_leaf2, P_leaf1_label0, P_leaf1_label1, P_leaf2_label0, P_leaf2_label1)
        if P_leaf1_label0==0.0 or P_leaf1_label1==0.0:
            HYL1=0
        else:
            HYL1=-1*(P_leaf1*(P_leaf1_label0* math.log(P_leaf1_label0, 2)) + P_leaf1*(P_leaf1_label1* math.log(P_leaf1_label1, 2)))

        if P_leaf2_label0==0.0 or P_leaf2_label1==0.0:
            HYL2=0
        else:
            HYL2=-1*(P_leaf2*(P_leaf2_label0* math.log(P_leaf2_label0, 2)) + P_leaf2*(P_leaf2_label1* math.log(P_leaf2_label1, 2)))

        HY=HYL1+HYL2
        ##print("TOTAL ENTROPY IS ", TotalEntropy)

        return TotalEntropy-HY



    # In[403]:


    def Spliting_Index(Training_data_l,features_left):
        MaxEn=-1
        #print("Left out features on which we need to split are ", features_left)
        for i in features_left:
            #print(i)
            MI=MutualInfo(Training_data_l,i)
            #print("#printing mutual info")
            #print(MI, i)
            if(MI>MaxEn):
                MaxEn=MI
                SplitFeature=i
        return [SplitFeature,MaxEn]
    def Base(Key,training_data_V):
        Key.isLeaf='Y'
       
        Assign_Key_Count(Key)
        Key.MajorityVote()





    # In[181]:





    # In[519]:


    def h(Key,training_data_V,features_left,depth,root1):
        #print("!!!!",len(training_data_V))
        ##print(depth,maxDepth)
        ##print(features_left)
        if ( (depth == maxDepth) or len(features_left)==0 or (depth >=maxDepth+1)):
            ##print(" I am in the base case ")
            Key.NodesData=training_data_V.copy()
            Base(Key,training_data_V)
            return
        else: 
            ##print("********** left over features are ", features_left)
            if(len(training_data_V) == 0):
                Key.NodesData="Null"
                Base(Key,training_data_V)
                return
            ##print(training_data_V)
            x=Spliting_Index(training_data_V,features_left)
            Mutual_Info=x[1]
            
            Key.node=x[0]
            ##print(Mutual_Info,Key.node)
            ##print("[DBG]: I am in H function call and split on")
            ##print(training_data_V)
            ##print(x[0])
            ##print(x[1], training_data_V[0], Key.node)
            #Split_In = int(training_data_V[0].index(Key.node))
            Split_In = int(train_data[0].index(Key.node))
            ##print("Splitting on ", Split_In)
            Node_Label_Count= CountLabel(training_data_V)
            ##print(Node_Label_Count)
            if Node_Label_Count[0]==0 or Node_Label_Count[1]==0:
                ##print("This is leaf node Boss")

                Key.isLeaf='Y'
                Assign_Key_Count(Key)
                Key.MajorityVote()
                Key.NodesData="Null"
                return
            LeftChild=[]
            RightChild=[]
            for row in training_data_V:
                if(row[Split_In]==leafValues[0]):
                    LeftChild.append(row)
                    #print(row)
                elif (row[Split_In]==leafValues[1]):
                    RightChild.append(row)
            features_left.remove(Key.node)
            #print("[DBG]LEFT CHILD AND RIGHT CHILD ARE")
            #print(LeftChild)
            #print(RightChild)
            Key.NodesData = LeftChild.copy() + RightChild.copy()
            selfLeft= Node("null")
            selfRight = Node("null")
            Key.LeftChild= selfLeft
            Key.RightChild= selfRight
            selfLeft.NodesData = LeftChild.copy()
            selfRight.NodesData = RightChild.copy()
            
            if(Mutual_Info>0.0):
                if Node_Label_Count[0]!= 0 and Node_Label_Count[1] != 0:
                    count=CountLabel(LeftChild)
                    strn=""
                    strn+= "|" *(depth)
                    strn+=" "
                    strn+=Key.node
                    strn+= " = "
                    strn+=leafValues[0]
                    strn += ": [" + str(count[0]) +" " + str(labels[0]) + "/"  + str(count[1]) +" " + str(labels[1])+ "]"
                    print(strn)
                    fl=features_left.copy()
                    #print("[DBG] <--------- LEFT CALL")
                    h(selfLeft,LeftChild,fl,depth+1,root1)    
                else:
                    Key.LeftChild.isLeaf='Y'
                #TODO: FIXME change the or to and
                if Node_Label_Count[0]!=0 and Node_Label_Count[1]!=0 :
                    count=CountLabel(RightChild)
                    strn=""
                    strn+= "|" *(depth)
                    strn+=" "
                    strn+=Key.node
                    strn+=" = "
                    strn+=leafValues[1]
                    strn += ": [" + str(count[0]) +" " + str(labels[0]) + "/"  + str(count[1]) + " " +str(labels[1])+ "]"
                    print(strn)
                    fr=features_left.copy()
                    #print("[DBG] ---------> RIGHT CALL")
                    h(selfRight,RightChild,fr,depth+1,root1)  
                else:
                    Key.RightChild.isLeaf='Y'
            else:
                Key.LeftChild.isLeaf = 'Y'
                Key.RightChild.isLeaf = 'Y'
            return root1





    # In[ ]:





    # In[413]:



    def predict(Key,row,predictData):
        ##print(row)
        if(Key.isLeaf=='Y'):
            ##print("Checking in the LEAF node");
            ##print(Key.classification)
            predictData.append(Key.classification)
            return
        else:
            ##print("Checking in the XXXXLEAF node");
            nodeName = Key.node
            ##print(nodeName, leafValues)
            ##print(features.index(nodeName))
            NodeIndex = int(features.index(nodeName))
            if( row[NodeIndex]== leafValues[0]):
                ##print("Traversing to left")
                ##print(Key.LeftChild)
                predict(Key.LeftChild,row,predictData)
            if(  row[NodeIndex]== leafValues[1]): 
                ##print("Traversing to Right")
                predict(Key.RightChild,row,predictData)



    # In[521]:


    root = Node("null")
    global maxDepth
    global TotalEntropy
    MaxEn=-1
    C1=0.0
    C2=0.0
    for i in range(1,len(train_data[1:])+1):
        #print(Training_Data_V[i][-1])
        row = train_data[i]
        if(row[-1] == labels[0]):
            C1 +=1;
        else:
            C2 +=1;
    p_label1= C1/(C1+C2);
    p_label2= C2/(C1+C2);
    TE=-1*(p_label1* math.log(p_label1, 2) + p_label2* math.log(p_label2, 2)) 
    TotalEntropy=TE
    ##print(TotalEntropy)
    #maxDepth=len(train_data[0])-1
    maxDepth=max_Depth+1
    temp = Node("null")
    #A#print("[DBG]\t I AM In TOP calling from root");
    Base_count=count=CountLabel(train_data)
    rr="[" + str(Base_count[0]) + " " +str(labels[0]) + "/"  + str(Base_count[1]) +" " + str(labels[1])+ "]"
    print(rr)
    h(root,data,features_left,1,root)


    # In[518]:

    queue = [];
    #print(" Now idiot, predict it")
    predictTrain=[]
    for i in range(1,len(data)):
        row=data[i]
        #print(row)
        predict(root,row,predictTrain);


# In[ ]:





# In[2]:


tdata=[]
with open(test_input, 'r') as tsv_file:                                                                                          
    tsv_read = csv.reader(tsv_file, delimiter="\t")
    for row in tsv_read:
        tdata.append(row)


# In[520]:

test_predict=[]
test_data=tdata
for i in range(1,len(test_data)):
    row=test_data[i]
    ##print(row)
    predict(root,row,test_predict);
#print(tdata)


# In[3]:


def ErrorOfOutput(original_data,predicted_data):
    error_count=0
    for i in range(1,len(original_data)):
        if original_data[i][-1] !=predicted_data[i-1]:
            #print("Hello")
            error_count=error_count+1
    error_result=error_count/(len(original_data)-1)
    #print(error_count)
    #print(error_result)
    return error_result


# In[ ]:





# In[4]:


train_error=ErrorOfOutput(train_data,predictTrain)
test_error= ErrorOfOutput(test_data,test_predict)
#TrEr.append(train_error)
#TeEr.append(test_error)


# In[ ]:





# In[5]:


with open(metric_out, 'w') as f_out:
    f_out.write("error(train): " + str(train_error))
    f_out.write('\n')
    f_out.write("error(test): " + str(test_error))


# In[6]:


with open(test_out, 'w') as f_out:
    for i in test_predict:
        f_out.write(i)
        f_out.write('\n')


# In[7]:


with open(train_out, 'w') as f_out:
    for i in predictTrain:
        f_out.write(i)
        f_out.write('\n')


# In[ ]:




