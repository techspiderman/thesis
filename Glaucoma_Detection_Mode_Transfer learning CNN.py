# Importing the Keras libraries and packages
import tensorflow as tf

import numpy as np
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tensorflow.keras.callbacks import History 
history = History()

# Load the Drive helper and mount
from google.colab import drive
# This will prompt for authorization.
drive.mount('/content/drive')

#### For Colab
test_path = "/content/drive/My Drive/datasets/riga"
test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path, target_size=(224,224), batch_size=32)

# ## Predict Fine_Tuned Model with ResNet50

#Approach 3 for prediction and labelling.We have used this approach.
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load model
model = load_model('/content/drive/My Drive/datasets/model2.h5')

#Creating test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        "/content/drive/My Drive/datasets/",
        target_size=(224, 224),
        batch_size=10,
        class_mode='binary',
        classes = ["rim_acrima_unlabelled"], 
        shuffle=False)

# Predict from generator (returns probabilities)
pred=model.predict_generator(test_generator, steps=len(test_generator), verbose=1)

#To convert the probabilities to class number
predicted_class_indices=np.argmax(pred,axis=1)
print(predicted_class_indices)

# Get classes by np.round
cl = np.round(pred)
# Get filenames (set shuffle=false in generator is important)
filenames=test_generator.filenames

# Data frame
resultsf=pd.DataFrame({"filenames":filenames,"normal_prob":np.round(pred[:,0],4), 
                      "glaucoma_prob":np.round(pred[:,1],4),
                      "class":predicted_class_indices})
resultsf

#Moving the images having >90% probability images to normal and glaucoma labelled folder.
import glob
import shutil
import os

src_dir = "./drive/My Drive/datasets"
dst_dir = "./drive/My Drive/datasets/rim_acrima_labelled/normal"
for index, file in resultsf.iterrows():
  #print(file)
  print(file["normal_prob"])
  if file["normal_prob"] >= 0.90:
    print(file["normal_prob"])
    folder_filename_l = file["filenames"].split("/")
    print(folder_filename_l[1])
    shutil.move(src_dir + "/" + folder_filename_l[0] +  "/" + folder_filename_l[1], dst_dir)
 
src_dirg = "./drive/My Drive/datasets"
dst_dirg = "./drive/My Drive/datasets/rim_acrima_labelled/glaucoma"
for index, file in resultsf.iterrows():
  print(file["glaucoma_prob"])
  if file["glaucoma_prob"] >= 0.90:
    print(file["glaucoma_prob"])
    folder_filename = file["filenames"].split("/")
    print(folder_filename[1])
    #print(src_dir + "/" + folder_filename[0] + "/" + folder_filename[1])
    shutil.move(src_dirg + "/" + folder_filename[0] + "/" + folder_filename[1], dst_dirg)

import os
#Check the number of images left in unlabelled folder.
directory = './drive/My Drive/datasets/riga'
print (len([item for item in os.listdir(directory)]))

#Check the number of images in labelled folder glaucoma.
import os
directory = './drive/My Drive/datasets/rim_acrima_labelled/glaucoma'
print (len([item for item in os.listdir(directory)]))
dir_original = './drive/My Drive/datasets/rim_acrima_labelled_original/glaucoma'
print(len([item for item in os.listdir(dir_original)]))

#Check the number of images in labelled folder normal.
directory = './drive/My Drive/datasets/rim_acrima_labelled/normal'
print (len([item for item in os.listdir(directory)]))
dir_original = './drive/My Drive/datasets/rim_acrima_labelled_original/normal'
print(len([item for item in os.listdir(dir_original)]))

# Creating the new resnet model with bigger data
import os 
import numpy as np
import shutil

# # Creating train / Val / test folders (One time use)
root_dir = './drive/My Drive/datasets/rim_acrima_labelled'
posCls = '/glaucoma'
negCls = '/normal'

if os.path.isdir(root_dir +'/sstrain' + posCls) == True:
  shutil.rmtree(root_dir +'/sstrain' + posCls)
os.makedirs(root_dir +'/sstrain' + posCls)
if os.path.isdir(root_dir +'/sstrain' + negCls) == True:
  shutil.rmtree(root_dir +'/sstrain' + negCls)
os.makedirs(root_dir +'/sstrain' + negCls)
if os.path.isdir(root_dir +'/ssval' + posCls) == True:
  shutil.rmtree(root_dir +'/ssval' + posCls)
os.makedirs(root_dir +'/ssval' + posCls)
if os.path.isdir(root_dir +'/ssval' + negCls) == True:
  shutil.rmtree(root_dir +'/ssval' + negCls)
os.makedirs(root_dir +'/ssval' + negCls)
if os.path.isdir(root_dir +'/sstest' + posCls) == True:
  shutil.rmtree(root_dir +'/sstest' + posCls)
os.makedirs(root_dir +'/sstest' + posCls)
if os.path.isdir(root_dir +'/sstest' + negCls) == True:
  shutil.rmtree(root_dir +'/sstest' + negCls)
os.makedirs(root_dir +'/sstest' + negCls)

# Creating partitions of the data after shuffeling
currentCls = posCls
src = "./drive/My Drive/datasets/rim_acrima_labelled"+currentCls # Folder to copy images from
print(src)
print(os.getcwd())
allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])

train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('testing: ', len(test_FileNames))
print(train_FileNames)


# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "./drive/My Drive/datasets/rim_acrima_labelled/sstrain"+currentCls)

for name in val_FileNames:
    shutil.copy(name, "./drive/My Drive/datasets/rim_acrima_labelled/ssval"+currentCls)

for name in test_FileNames:
    shutil.copy(name, "./drive/My Drive/datasets/rim_acrima_labelled/sstest"+currentCls)

# Creating partitions of the normal data after shuffeling

normalCls = negCls
src_n = "./drive/My Drive/datasets/rim_acrima_labelled"+normalCls # Folder to copy images from
print(src)
print(os.getcwd())
allFileNames_n = os.listdir(src_n)
np.random.shuffle(allFileNames_n)
train_FileNames_n, val_FileNames_n, test_FileNames_n = np.split(np.array(allFileNames_n),
                                                          [int(len(allFileNames_n)*0.7), int(len(allFileNames_n)*0.85)])


train_FileNames_n = [src_n+'/'+ name for name in train_FileNames_n.tolist()]
val_FileNames_n = [src_n+'/' + name for name in val_FileNames_n.tolist()]
test_FileNames_n = [src_n+'/' + name for name in test_FileNames_n.tolist()]

print('Total images: ', len(allFileNames_n))
print('Training: ', len(train_FileNames_n))
print('Validation: ', len(val_FileNames_n))
print('Testing: ', len(test_FileNames_n))

# Copy-pasting images
for name in train_FileNames_n:
    shutil.copy(name, "./drive/My Drive/datasets/rim_acrima_labelled/sstrain"+ normalCls)

for name in val_FileNames_n:
    shutil.copy(name, "./drive/My Drive/datasets/rim_acrima_labelled/ssval"+ normalCls)

for name in test_FileNames_n:
    shutil.copy(name, "./drive/My Drive/datasets/rim_acrima_labelled/sstest"+ normalCls)


#Creating the new resnet model with the bigger data
#### For Colab
train_path = "/content/drive/My Drive/datasets/rim_acrima_labelled/sstrain"
test_path = "/content/drive/My Drive/datasets/rim_acrima_labelled/sstest"
valid_path = "/content/drive/My Drive/datasets/rim_acrima_labelled/ssval"

train_batches = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, 
       height_shift_range=0.1, shear_range=0.15, zoom_range=0.1, 
       channel_shift_range=10., horizontal_flip=True).flow_from_directory(train_path, target_size=(224,224), classes =['normal','glaucoma'], batch_size=10)
test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path, target_size=(224,224), classes =['normal','glaucoma'], batch_size=10)
valid_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(valid_path, target_size=(224,224), classes =['normal','glaucoma'], batch_size=10)

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
from tensorflow.keras import applications
img_height,img_width = 224,224 
num_classes = 2

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

type(model)
model.summary()
history = model.fit_generator(train_batches, 
                    steps_per_epoch=30, 
                    validation_data= valid_batches, 
                    validation_steps=30,
                    epochs=20,
                    verbose=2)
# Model saved to a single file
from numpy import loadtxt

# save model and architecture to single file as model3 prepared from bigger data.
model.save("/content/drive/My Drive/datasets/ssmodel.h5")
print("Saved model to disk")
#Load a Keras Model
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model

# load model
model = load_model('/content/drive/My Drive/datasets/ssmodel.h5')
# summarize model.
model.summary()


# **Predict Fine_Tuned Model with ResNet50**


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
print(x_labels)
plots(test_imgs, titles=test_labels)
predictions = model.predict_generator(test_batches_fine, steps=1, verbose=0)
y_labels = predictions.argmax(axis=1)
y_labels

cm = confusion_matrix(x_labels, y_labels)
cm_plot_labels = ['Normal','Glaucoma']
plot_confusion_matrix(cm, cm_plot_labels, title ='Confusion Matrix')

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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

# **Calculating AUC** 
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
