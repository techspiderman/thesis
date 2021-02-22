
# model averaging ensemble
from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy
from numpy import array
from numpy import argmax
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

## Importing the necessary libraries"""

# Commented out IPython magic to ensure Python compatibility.
# Importing the Keras libraries and packages
import numpy as np
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.python.framework.ops import Tensor
from typing import Tuple, List
import glob
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
# %matplotlib inline

# Load the Drive helper and mount
from google.colab import drive
# This will prompt for authorization.
drive.mount('/content/drive')

#### For Colab
test_path = "/content/drive/My Drive/datasets/rim_acrima_labelled/test"

#test images 
test_batches_fine = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path, target_size=(224,224), classes =['normal','glaucoma'], batch_size=20)

"""# Loading the models"""

#from keras.models import load_model

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# image folder

# path to model
model1_path = "/content/drive/My Drive/datasets/model2.h5"
model2_path = "/content/drive/My Drive/datasets/ss_model.h5"
model3_path = "/content/drive/My Drive/datasets/dcgan_model.h5"
# dimensions of images
img_width, img_height = 224, 224

# load the trained model
model1 = load_model(model1_path)
model2 = load_model(model2_path)
model3 = load_model(model3_path)

models = [model1, model2, model3]


"""# Model 1 Evaluation"""

#Make predictions for model1
predictions1 = model1.predict(test_batches_fine, steps = 1, verbose=0)

np.round(predictions1, 3)

test_imgs, test_labels = next(test_batches_fine)
x_labels = test_labels.argmax(axis=1)
x_labels

y_labels1 = predictions1.argmax(axis=1)
y_labels1

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

cm = confusion_matrix(x_labels, y_labels1)
cm_plot_labels = ['Normal','Glaucoma']
plot_confusion_matrix(cm, cm_plot_labels, title ='Confusion Matrix')

#Calculating AUC and accuracy score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
print(x_labels.shape)
print(predictions1[:,0].shape)
score = accuracy_score(x_labels, y_labels1)
print(score)
fpr, tpr,threshold = roc_curve(x_labels, predictions1[:,1])
roc_auc = auc(fpr, tpr)
#print (fpr)
#print(tpr)
print(roc_auc)

#Calculating and Plotting the AUC.
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
#probs = resnet_model.predict_proba(x_labels)
preds = predictions1[:,1]
fpr, tpr, threshold = metrics.roc_curve(x_labels, preds)
roc_auc = metrics.auc(fpr, tpr)
#print (fpr)
#print (tpr)
print (roc_auc)
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

#Calculating precision, recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(x_labels, y_labels1))

"""# Model 2 Evaluation"""

predictions2 = model2.predict(test_batches_fine, steps = 1, verbose=0)

np.round(predictions2, 3)

y_labels2 = predictions2.argmax(axis=1)
y_labels2

cm = confusion_matrix(x_labels, y_labels2)
cm_plot_labels = ['Normal','Glaucoma']
plot_confusion_matrix(cm, cm_plot_labels, title ='Confusion Matrix')

#Calculating AUC and accuracy score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
print(x_labels.shape)
print(predictions2[:,0].shape)
score = accuracy_score(x_labels, y_labels2)
print(score)
fpr, tpr,threshold = roc_curve(x_labels, predictions2[:,1])
roc_auc = auc(fpr, tpr)
#print (fpr)
#print(tpr)
print(roc_auc)

#Calculating and Plotting the AUC.
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
#probs = resnet_model.predict_proba(x_labels)
preds = predictions2[:,1]
fpr, tpr, threshold = metrics.roc_curve(x_labels, preds)
roc_auc = metrics.auc(fpr, tpr)
#print (fpr)
#print (tpr)
print (roc_auc)
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

#Calculating precision, recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(x_labels, y_labels2))

"""#Model 3 Evaluation"""

predictions3 = model3.predict(test_batches_fine, steps = 1, verbose=0)

y_labels3 = predictions3.argmax(axis=1)
y_labels3

cm = confusion_matrix(x_labels, y_labels3)
cm_plot_labels = ['Normal','Glaucoma']
plot_confusion_matrix(cm, cm_plot_labels, title ='Confusion Matrix')

#Calculating AUC and accuracy score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
print(x_labels.shape)
print(predictions3[:,0].shape)
score = accuracy_score(x_labels, y_labels3)
print(score)
fpr, tpr,threshold = roc_curve(x_labels, predictions3[:,1])
roc_auc = auc(fpr, tpr)
#print (fpr)
#print(tpr)
print(roc_auc)

#Calculating and Plotting the AUC.
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
#probs = resnet_model.predict_proba(x_labels)
preds = predictions3[:,1]
fpr, tpr, threshold = metrics.roc_curve(x_labels, preds)
roc_auc = metrics.auc(fpr, tpr)
#print (fpr)
#print (tpr)
print (roc_auc)
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

#Calculating precision, recall, F1 score
from sklearn.metrics import classification_report
print(classification_report(x_labels, y_labels3))

"""# Ensemble Evaluation"""

#Create an ensemble by taking an average of 3 models.
ensemble_predictions = (predictions1 + predictions2 + predictions3)/3

test_imgs, test_labels = next(test_batches_fine)

x_labels = test_labels.argmax(axis=1)
x_labels

y_labels = ensemble_predictions.argmax(axis=1)
y_labels

cm = confusion_matrix(x_labels, y_labels)
cm_plot_labels = ['Normal','Glaucoma']
plot_confusion_matrix(cm, cm_plot_labels, title ='Confusion Matrix')

#Calculating AUC and accuracy score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
print(x_labels.shape)
print(ensemble_predictions[:,0].shape)
score = accuracy_score(x_labels, y_labels)
print(score)
fpr, tpr,threshold = roc_curve(x_labels, ensemble_predictions[:,1])
roc_auc = auc(fpr, tpr)
#print (fpr)
#print(tpr)
print(roc_auc)

#Calculating and Plotting the AUC.
import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
#probs = resnet_model.predict_proba(x_labels)
preds = ensemble_predictions[:,1]
fpr, tpr, threshold = metrics.roc_curve(x_labels, preds)
roc_auc = metrics.auc(fpr, tpr)
#print (fpr)
#print (tpr)
print (roc_auc)
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

"""# calculate precision, recall and F1 score."""
from sklearn.metrics import classification_report
print(classification_report(x_labels, y_labels))
