
# coding: utf-8

# In[1]:


import sys
get_ipython().system(u'{sys.executable} -m pip install numpy')


# In[2]:


import numpy


# In[3]:


import keras


# In[4]:


import sys
get_ipython().system(u'{sys.executable} -m pip install keras')


# In[5]:


import keras


# In[6]:


import sys
get_ipython().system(u'{sys.executable} -m pip install tensorflow')


# In[7]:


import keras


# In[9]:


import numpy
import keras
from keras import backend as K
from keras.models import Sequential 
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt


# In[10]:


import sys
get_ipython().system(u'{sys.executable} -m pip install matplotlib')


# In[12]:


import sys
python -m pip install --upgrade pip


# In[13]:


import sys
get_ipython().system(u'{sys.executable} -m pip install sklearn.metrics')


# In[14]:


import sys
get_ipython().system(u'{sys.executable} -m pip install sklearn')


# In[15]:


import sys
get_ipython().system(u'{sys.executable} -m pip install itertools')


# In[3]:


import numpy
import keras
from keras import backend as K
from keras.models import Sequential 
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import os



train_path='Desktopmlmp\train'
test_path='mlmp\test'
valid_path='mlmp\valid'
train_datagen=ImageDataGenerator()


train_batches=train_datagen.flow_from_directory(directory=train_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)
test_batches= train_datagen.flow_from_directory(directory=test_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)
valid_batches= train_datagen.flow_from_directory(directory= valid_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)


# In[19]:


train_path='mlmp\train'
test_path='mlmp\test'
valid_path='mlmp\valid'


# In[48]:


from keras.preprocessing.image import ImageDataGenerator

train_path='mlmp\train'
test_path='mlmp\test'
valid_path='mlmp\valid'

train_datagen=ImageDataGenerator()



train_batches=train_datagen.flow_from_directory(train_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)
test_batches=  train_datagen.flow_from_directory(test_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)
valid_batches= train_datagen.flow_from_directory(valid_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)


# In[34]:


from keras.preprocessing.image import ImageDataGenerator

train_path='mlmp\train'
train_batches=ImageDataGenerator.flow_from_directory(directory=train_path, target_size=(244, 244), color_mode='grayscale', classes=['male','female'], batch_size=10)


# In[59]:


import os
train_path=r'C:\Users\Lenovo\Desktop\mlmp\train'
test_path=r'C:\Users\Lenovo\Desktop\mlmp\test'
valid_path=r'C:\Users\Lenovo\Desktop\mlmp\valid'

train_datagen=ImageDataGenerator()



train_batches=train_datagen.flow_from_directory(train_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)
test_batches=  train_datagen.flow_from_directory(test_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)
valid_batches= train_datagen.flow_from_directory(valid_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)



# In[66]:


# plots images with labels within jupyter notebook
import numpy as np
import matplotlib.pyplot as plt

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
imgs,labels=next(train_batches)
plots(imgs,titles=labels)


# In[8]:


#Build and train cnn

model =Sequential ([Conv2D(32,(3,3), activation='relu',input_shape=(224,224,3)),
                    Flatten(),
                    Dense(2,activation='softmax'),
                   ])


# In[9]:


model.compile(Adam(lr=.0001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[10]:


model.fit_generator(train_batches,steps_per_epoch=10,validation_data=valid_batches,validation_steps=10,epochs=5,verbose=2)


# In[5]:


import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

train_path=r'C:\Users\Lenovo\Desktop\mlmp\train'
test_path=r'C:\Users\Lenovo\Desktop\mlmp\test'
valid_path=r'C:\Users\Lenovo\Desktop\mlmp\valid'

train_datagen=ImageDataGenerator()



train_batches=train_datagen.flow_from_directory(train_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)
test_batches=  train_datagen.flow_from_directory(test_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)
valid_batches= train_datagen.flow_from_directory(valid_path,target_size=(224,224),classes=None,class_mode='categorical',batch_size=10)


def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
imgs,labels=next(train_batches)
plots(imgs,titles=labels)


test_imgs,test_labels= next (test_batches)
plots(test_imgs,titles=test_labels)


# In[6]:


#for male we have 0 label for female we have 1
test_labels=test_labels[:,0]
test_labels


# In[14]:


predictions=model.predict_generator(test_batches,steps=1,verbose=0)


# In[15]:


predictions


# In[16]:


cm=confusion_matrix(test_labels,predictions[:,0])


# In[21]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    # Only use the labels that appear in the data
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment="center",color="white" if cm[i,j]>thresh else "black")
        
    plt.tight_layout()
    plt.xlabel('Predicted Label')
    plt.ylabel('True label')


# In[22]:


cm_plot_labels=['male','female']
plot_confusion_matrix(cm,cm_plot_labels,title='Confusion Matrix')


# In[ ]:


#we are going to finetune the existing model 
vgg16_model=keras.applications.vgg16.VGG16()

