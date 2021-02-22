# Importing the Keras libraries and packages
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.callbacks import History 
history = History()


# Load the Drive helper and mount
from google.colab import drive
# This will prompt for authorization.
drive.mount('/content/drive')


#### For Colab
train_path = "/content/drive/My Drive/datasets/rim_acrima_labelled/train"
test_path = "/content/drive/My Drive/datasets/rim_acrima_labelled/test"
valid_path = "/content/drive/My Drive/datasets/rim_acrima_labelled/val"

                                    
train_batches = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, 
       height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, 
       channel_shift_range=10., horizontal_flip=True, vertical_flip=True).flow_from_directory(train_path, target_size=(224,224), classes =['normal','glaucoma'], batch_size=32)
test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path, target_size=(224,224), classes =['normal','glaucoma'], batch_size=32)
valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path, target_size=(224,224), classes =['normal','glaucoma'], batch_size=32)

def plot_confusion_matrix(cm, classes, 
                          normalize=False,
                         title = 'Confusion Matrix',
                         cmap=plt.cm.Blues):
  
  plt.imshow(cm, interpolation = 'nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation = 45)
  plt.yticks(tick_marks, classes)
  
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]
    print("Normalized Confusion Matrix")
  else:
    print("Confusion Matrix without normalization")
  
  print(cm)
  
  thresh = cm.max() / 2.
  
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j,i, cm[i,j],
            horizontalalignment ="center",
            color = "white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
  

import tensorflow as tf
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
from urllib.request import urlopen,urlretrieve
from PIL import Image
from tqdm import tqdm_notebook
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
import cv2
import tensorflow.keras as keras

#from resnets_utils import *

from tensorflow.keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from tensorflow.keras import backend


img_height,img_width = 224,224 
num_classes = 2
#If imagenet weights are being loaded, 
#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
resnet_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (img_height,img_width,3))

x = resnet_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = resnet_model.input, outputs = predictions)


from tensorflow.keras.optimizers import SGD, Adam
#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])

type(resnet_model)
model.summary()

from tensorflow.keras.utils import plot_model
#Plot the the architecture
plot_model(model, to_file='/content/drive/My Drive/datasets/rim_acrima_labelled/resnet_plot.png', show_shapes=True, show_layer_names=True)


# ## train Fine_Tuned Model with ResNet50

history = model.fit_generator(train_batches, 
                    steps_per_epoch=20, 
                    validation_data= valid_batches, 
                    validation_steps=20,
                    epochs=100,
                    verbose=2)


# Plot Model Loss and Model accuracy
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])  # RAISE ERROR
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']) #RAISE ERROR
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()





# **bold text**## **Predict Fine_Tuned Model with ResNet50**



def plots(ims, figsize=(12,6), rows=5, interp=False, titles=None):
  if type(ims[0]) is np.ndarray:
    ims = np.array(ims).astype(np.uint8)
    if(ims.shape[-1] != 3):
      ims = ims.transpose ((0,2,3,1))
  f = plt.figure(figsize = figsize)
  cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows +1
  for i in range(len(ims)):
    sp = f.add_subplot (rows, cols, i+1)
    sp.axis('Off')
    if titles is not None:
      sp.set_title(titles[i], fontsize =10)
    plt.imshow(ims[i], interpolation=None if interp else 'none')
    
#test_batches_fine = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path, target_size=(224,224), classes =['normal','glaucoma'], batch_size=50)
test_batches_fine = test_batches


test_imgs, test_labels = next(test_batches_fine)



x_labels = test_labels.argmax(axis=1)
x_labels

predictions = model.predict(test_batches_fine, steps = 1, verbose=0)
print(predictions)





probs_test = []
for i, prob in enumerate(y_probabilities):
  if prob[0] > 0.5:
    probs_test.append(1)
  else:
    print(probs_test)
    
score = accuracy_score(x_labels, y_labels)
fpr, tpr, _ = roc_curve(x_labels, y_probabilities[:,0])
auc_score = auc(fpr, tpr)

print(auc_score)

cm = confusion_matrix(x_labels, y_labels)


cm_plot_labels = ['Normal','Glaucoma']


plot_confusion_matrix(cm, cm_plot_labels, title ='Confusion Matrix')


# Calculating AUC
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
print(x_labels.shape)
print(predictions[:,0].shape)
score = accuracy_score(x_labels, y_labels)
print(score)
fpr, tpr,threshold = roc_curve(x_labels, predictions[:,0])
roc_auc = auc(fpr, tpr)

print(roc_auc)



#Plot of a ROC curve for a specific class

plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()




import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = resnet_model.predict_proba(x_labels)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()



import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# **Save the model to reuse it later**
get_ipython().system('pip install h5py')


#  Save a Keras Model

# Model saved to a single file
from numpy import loadtxt

# save model and architecture to single file
model.save("/content/drive/My Drive/datasets/finalcnn_model.h5")
print("Saved model to disk")
#Load a Keras Model
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model

# load model
resnet_model = load_model('/content/drive/My Drive/datasets/finalcnn_model.h5')
# summarize model.
resnet_model.summary()


