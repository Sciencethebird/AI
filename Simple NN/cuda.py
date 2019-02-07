


# coding: utf-8

# ## Import the library

# In[2]:


import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn


# ## layer definition (Need to do!!!)

# In[3]:


def InnerProduct_For(x,W,b):
    y = cp.dot(x,W)+b
    return y

def InnerProduct_Back(dEdy,x,W,b):
    dEdx = cp.dot( dEdy, W.transpose() )
    dEdW = cp.dot(x.transpose(), dEdy)
    dEdb = dEdy 
    
    return dEdx, dEdW, dEdb

def Softmax_For(x):
    exps = np.exp(x)
    exps_sum = cp.sum(exps, axis = 1)
    exps_sum = exps_sum.reshape(exps[:, 0].size, 1)
    softmax = exps / exps_sum# bug fixed here 
    return softmax

def Softmax_Back(y,t):
    dEdx = y - t
    return dEdx

def Sigmoid_For(x):
    return float(1.0)/( 1+ np.exp(-x) ) #change here int 

def Sigmoid_Back(dEdy ,x):

    dEdx = np.exp(-x) / np.square( 1+np.exp(-x)  )  *  dEdy
    return dEdx

def loss_For(y,y_pred):

    loss =  - cp.sum(  y * np.log( y_pred )  ) 
    return loss
'''
def ReLu_For(x):

    return y

def ReLu_Back(dEdy,x):
            
    return dEdx
'''



# ## Setup the Parameters and Variables (Can tune that!!!)

# In[4]:


eta =  0.00005      #learning rate       ###########lolololololol
Data_num = 784      #size of input data   (inputlayer, 784 pixels)
W1_num = 15         #size of first neural (1st hidden layer)
Out_num = 10        #size of output data  (output layer)
iteration = 1000       #epoch for training   (iteration)
image_num = 60000   #input images
test_num  = 10000   #testing images

## Cross Validation ##
##spilt the training data to 80% train and 20% valid##
train_num = int(image_num*0.8)
valid_num = int(image_num*0.2)


# ## Setup the Data (Create weight array here!!!)

# In[5]:


w_1= (np.random.normal(0,0.1,Data_num*W1_num)).reshape(Data_num,W1_num)
w_out  = (np.random.normal(0,0.1,W1_num*Out_num)).reshape(W1_num, Out_num)
b_1, b_out = randn(1,W1_num),randn(1,Out_num)
print("w1 shape:", w_1.shape)
print("w_out shape:", w_out.shape)
print("b_1 shape:", b_1.shape)
print("b_out shape:", b_out.shape)
print(w_1)


# ## Prepare all the data

# ### Load the training data and labels from files

# In[6]:


df = pd.read_csv('fashion-mnist_train_data.csv')
fmnist_train_images = df.values
print("Training data:",fmnist_train_images.shape[0])
print("Training data shape:",fmnist_train_images.shape)

df = pd.read_csv('fashion-mnist_test_data.csv')
fmnist_test_images = df.values
print("Testing data:",fmnist_test_images.shape[0])
print("Testing data shape:",fmnist_test_images.shape)

df = pd.read_csv('fashion-mnist_train_label.csv')
fmnist_train_label = df.values
print("Training labels shape:",fmnist_train_label.shape)



# ### Show the 100 testing images

# In[7]:

'''
plt.figure(figsize=(20,20))
for index in range(100):
    image = fmnist_test_images[index].reshape(28,28)
    plt.subplot(10,10,index+1)
    plt.imshow(image)
plt.show() 

'''
# ### Convert the training labels data type to one hot type

# In[8]:


label_temp = np.zeros((image_num,10), dtype = np.float32)
for i in range(image_num):
    label_temp[i][fmnist_train_label[i][0]] = 1
train_labels_onehot = np.copy(label_temp)
print("\nTraining labels shape:",train_labels_onehot.shape)


# ### Separate train_images, train_labels into training and validating 

# In[13]:

train_data_img = np.copy(fmnist_train_images[:train_num,:])
train_data_lab = np.copy(train_labels_onehot[:train_num,:])
valid_data_img = np.copy(fmnist_train_images[train_num:,:])
valid_data_lab = np.copy(train_labels_onehot[train_num:,:])

# Normalize the input data between (0,1)
train_data_img = train_data_img/255.
valid_data_img = valid_data_img/255.
test_data_img = fmnist_test_images/255.

print("Train images shape:",train_data_img.shape)
print("Train labels shape:",train_data_lab.shape)
print("Valid images shape:",valid_data_img.shape)
print("Valid labels shape:",valid_data_lab.shape)
print("Test  images shape:",test_data_img.shape)
print(train_data_lab)


# ## Execute the Iteration (Need to do!!!)

# In[12]:


valid_accuracy = [0]

for itr in range(iteration):
        
        if itr % 10 == 0 : 
            print(']n', itr / iteration * 100 ,'% done\n')
    # Forward-propagation
        img  = train_data_img
        w1   = w_1
        b1   = b_1
        wout = w_out
        bout = b_out
   
        z1 = InnerProduct_For(img, w1, b1) # save for sigmoid prime (backpropagation)
        zout = Sigmoid_For(z1)

        #Softmax
        yp = Softmax_For(InnerProduct_For(zout, wout, bout)) 
       

        #Cross Entropy Loss
        loss = loss_For(train_data_lab, yp) 
        print('\n\nLoss', loss)
        #valid_accuracy.append(loss)

    # Bakcward-propagation
  
        soft_back = Softmax_Back(yp, train_data_lab) # y-t

        #output layer
        dEdx, dEdW, dEdb = InnerProduct_Back(soft_back, zout, w_out, b_out) #InnerProduct_Back(dEdy,x,W,b)
        bout_grad = dEdb #np.average(dEdb, axis = 0)
        wout_grad = dEdW #np.average(dEdW, axis = 0)
        #input layer
        dEdx, dEdW, dEdb = InnerProduct_Back(Sigmoid_Back(dEdx , z1), img, w_1, b_1)
        b1_grad = dEdb #np.average(dEdb, axis = 0)
        w1_grad = dEdW #np.average(dEdW, axis = 0)
  

    # Parameters Updating (Gradient descent)
        w_out = w_out - eta * wout_grad
        b_out = np.average(b_out - eta * bout_grad, axis = 0)
        w_1 = w_1 - eta * w1_grad
        b_1 = np.average(b_1 - eta * b1_grad, axis = 0)
        
    

    # Do cross-validation to evaluate model
        valid_img = valid_data_img
     
        zout = Sigmoid_For(InnerProduct_For(valid_img, w1, b1))
        #Softmax
        yp = Softmax_For(InnerProduct_For(zout, wout, bout)) 
       
    # Get 1-D Prediction array
        count = 0
        for i in range(valid_num):
           
            if np.argmax(yp[i,:]) == np.argmax(valid_data_lab[i, :]):
                #print(i, yp[i,:], valid_data_lab[i][:] )
                #print(i, np.argmax(yp[i,:]), np.argmax(valid_data_lab[i,:]) )
                count = count + 1

        accuracy = count/valid_num*100
        #print('Correct: ', count,'/', valid_num, '\n', float(accuracy), '%')
    # Compare the Prediction and validation
    
    # Calculate the accuracy

        print (itr, 'th,', 'eta = ', eta)
        if itr > 70 and  eta > 0.00001 and accuracy - valid_accuracy[-1] < -0.3 :
            eta = eta / 1.1
            print('eta changed to ', eta)
        if accuracy - valid_accuracy[-1]< 0.01 and accuracy - valid_accuracy[-1] > 0 and itr>300:
            eta = eta * 1.01
            print('eta changed to ', eta)
        #if itr >500:
            #eta = 0.000005
        valid_accuracy.append(accuracy)
        print('Correct: ', count,'/', valid_num, '\n', float(accuracy), '%')

'''       
print(valid_accuracy)     
plt.plot(np.arange(0, iteration, 1), valid_accuracy, label = 'accuracy')
plt.show()
'''
'''
# ## Testing Stage

# ### Predict the test images (Do forward propagation again!!!)

# In[10]:


# Forward-propagation


# ### Convert results to csv file (Input the (10000,10) result array!!!)

# In[12]:


# Convert "test_Out_data" (shape: 10000,10) to "test_Prediction" (shape: 10000,1)

test_Prediction      = np.argmax(test_Out_data, axis=1)[:,np.newaxis].reshape(test_num,1)
df = pd.DataFrame(test_Prediction,columns=["Prediction"])
df.to_csv("DL_LAB1_prediction_ID.csv",index=True, index_label="index")



# ## Convert results to csv file

# In[16]:


accuracy = np.array(valid_accuracy)
plt.plot(accuracy, label="$iter-accuracy$")
y_ticks = np.linspace(0, 100, 11)
plt.legend(loc='best')
plt.xlabel('iteration')
plt.axis([0, iteration, 0, 100])
plt.ylabel('accuracy')
plt.show()
'''
