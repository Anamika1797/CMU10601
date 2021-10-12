#!/usr/bin/env python
# coding: utf-8

# In[19]:


import sys
import numpy as np
import csv

if __name__ == "__main__":     
    
    data = []
    
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    split_index = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metric_out = sys.argv[6]
    print("The input file is: %s" % (train_input))
    print("The input file is: %s" % (test_input))
    print("The input file is: %s" % (split_index))
    print("The output file is: %s" % (train_out))
    print("The output file is: %s" % (test_out))
    print("The output file is: %s" % (metric_out))
    
    no_of_feature=split_index
    
    with open(train_input) as tsv_file:                                                                                          
        tsv_read = csv.reader(data, delimiter="\t")
        for row in tsv_read:
            data.append(row)


# In[183]:


import sys
import numpy as np
import csv
'''
train_input = "C:\\Users\\Anamika Shekhar\\handout\\education_train.tsv"
test_input = "C:\\Users\\Anamika Shekhar/handout/education_test.tsv"
split_index = 5
train_out = "C:\\Users\\Anamika Shekhar/handout/output/train.labels"
test_out = "C:\\Users\\Anamika Shekhar/handout/output/test.labels"
metric_out = "C:\\Users\\Anamika Shekhar\handout\output\metrics.txt"
''' 


# Import data and Load the data into array and separate the spliting feature and output label

# In[184]:


data = []

no_of_feature=split_index
    
with open(train_input, 'r') as tsv_file:                                                                                          
    tsv_read = csv.reader(tsv_file, delimiter="\t")
    for row in tsv_read:
        data.append(row)


# In[185]:


import numpy as np
training_data=np.asarray(data).transpose()
root_node=training_data[no_of_feature,1:]
y_train = training_data[np.shape(training_data)[0] -1] [1:]  


# In[186]:


##print(y_train)


# In[187]:


##print(root_node)


# In[188]:


##print(y_train)


# Split the data of the root node into yes and no leaf and then assign the output labels to variable

# In[189]:



if (training_data[len(training_data)-1][0] == 'grade'):
    yes_leaf = np.asarray(np.where(root_node=='A'))
    no_leaf = np.where(root_node=='notA')
if (training_data[len(training_data)-1][0] == 'Party'):
    yes_leaf = np.where(root_node=='y')     
    no_leaf = np.where(root_node=='n')
        
if (training_data[len(training_data)-1][0] == 'Party'):
    class1= 'republican'
    class2 = 'democrat'
if (training_data[len(training_data)-1][0] == 'grade'):
    class1 = 'A'
    class2 = 'notA'


# In[190]:


##len(no_leaf[0])


# In[191]:


label_yesList = y_train[yes_leaf]
label_noList = y_train[no_leaf]
##print(y_train)


# Find the count of the two classifications in yes and no leaf and then find the majority vote

# In[192]:


def MajorityVote(output_list_Y,output_list_N,class1,class2):
    class1_list_yes=np.where(output_list_Y==class1)
    count_of_class1_yes=(len(class1_list_yes[0]))
    class1_list_no=np.asarray(np.where(output_list_N==class1))
    count_of_class1_no=(len(class1_list_no[0]))
    class2_list_yes=np.asarray(np.where(output_list_Y==class2))
    count_of_class2_yes=(len(class2_list_yes[0]))
    class2_list_no=np.asarray(np.where(output_list_N==class2))
    count_of_class2_no=(len(class2_list_no[0]))
    ##print(class1_list_yes)
    #print(class2_list_yes)
    ##print(class1_list_no)
    ##print(class2_list_no)
    if(count_of_class1_yes>count_of_class2_yes):
        Vote_Y_leaf=class1
    elif(count_of_class1_yes<count_of_class2_yes):
        Vote_Y_leaf=class2
    else:
        Vote_Y_leaf=class2
    if(count_of_class1_no>count_of_class2_no):
        Vote_N_leaf=class1
    elif(count_of_class1_no<count_of_class2_no):
        Vote_N_leaf=class2
    else:
        Vote_N_leaf=class2
    ##print(Vote_Y_leaf)
    ##print(Vote_N_leaf)
    return Vote_Y_leaf,Vote_N_leaf


# In[ ]:


##print(count_of_class2_yes)
##print(count_of_class2_no)


# In[ ]:


##print(Vote_Y_leaf)


# In[ ]:





# In[196]:


def OutputHypothesis(y_train,test_input,no_of_feature,Vote_Y_leaf,Vote_N_leaf):
    data_test=[]
    with open(test_input, 'r') as tsv_file:                                                                                          
        tsv_read = csv.reader(tsv_file, delimiter="\t")
        for row in tsv_read:
            data_test.append(row)
    test_data=np.asarray(data_test).transpose()  
    root_node_test=test_data[no_of_feature,1:]
    y_test = test_data[np.shape(test_data)[0] -1] [1:]
    ##print(y_test)
    ##len(data_test)
    if (test_data[len(test_data)-1][0] == 'grade'):
        yes_leaf_test = np.where(root_node_test=='A')
        no_leaf_test = np.where(root_node_test=='notA')
    if (test_data[len(test_data)-1][0] == 'Party'):
        yes_leaf_test = np.where(root_node_test=='y')      
        no_leaf_test = np.where(root_node_test=='n')
    ##print(y_train)
        
    predict_output_train=np.copy(y_train)
    predict_output_train[yes_leaf] =  Vote_Y_leaf    
    predict_output_train[no_leaf] =  Vote_N_leaf  
    predict_output_test=np.copy(y_test)
    predict_output_test[yes_leaf_test] =  Vote_Y_leaf    
    predict_output_test[no_leaf_test] =  Vote_N_leaf 
    ##print(predict_output_train)
    ##print(y_train)
    return predict_output_train,predict_output_test,data_test,y_test,y_train


# In[ ]:





# In[ ]:



##print(test_data)


# In[ ]:





# In[ ]:


##print(y_test)


# In[ ]:


##predict_output_test[1]


# In[194]:



def ErrorOfOutput(data,data_test,predict_output_train,predict_output_test,y_train,y_test):
    test_correct_count=0
    correct_predict_test=np.where(test_output==y_test)

    test_error=1-len(correct_predict_test[0])/(len(data_test)-1)

    correct_count_train=0
    correct_predict__train=np.where(train_output==y_train)

    train_error=1-len(correct_predict__train[0])/(len(data)-1)
    ##print(train_error)
   ## print(test_error)
    
    return test_error,train_error


# In[15]:





# In[197]:


VoteY,VoteN=MajorityVote(label_yesList,label_noList,class1,class2)
train_output,test_output,test_data,y_test,y_train=OutputHypothesis(y_train,test_input,no_of_feature,VoteY,VoteN)
test_error,train_error=ErrorOfOutput(data,test_data,train_output,test_output,y_train,y_test)


# In[ ]:





# In[ ]:





# In[200]:





# In[ ]:





# In[204]:


with open(metric_out, 'w') as f_out:
    f_out.write("error(train): " + str(train_error))
    f_out.write('\n')
    f_out.write("error(test): " + str(test_error))


# In[205]:


with open(test_out, 'w') as f_out:
    for i in test_output:
        f_out.write(i)
        f_out.write('\n')


# In[206]:


with open(train_out, 'w') as f_out:
    for i in train_output:
        f_out.write(i)
        f_out.write('\n')


# In[ ]:




