#!/usr/bin/env python
# coding: utf-8

# In[555]:


import keras
import tensorflow as tf
#import vis ## keras-vis
import matplotlib.pyplot as plt
import numpy as np
print("keras      {}".format(keras.__version__))
print("tensorflow {}".format(tf.__version__))


# In[556]:


from keras.applications.vgg16 import VGG16, preprocess_input
model = VGG16(weights='imagenet')
model.summary()
for ilayer, layer in enumerate(model.layers):
    print("{:3.0f} {:10}".format(ilayer, layer.name))


# In[560]:


from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

n = 0
#img = Image.open('./selfie2anime/testA/female_10328.jpg')
#dim = (255,255,3)
path = './selfie2anime/testB'
img_show = np.zeros([100,256,256,3])
for path,dir,file in os.walk(path):
    for img_get in file:
        img_open = Image.open(path+'/'+img_get)
        #plt.imshow(img_open)
        img_array = np.array(img_open)
        #img_resize = cv2.resize(img_array,dim,interpolation = cv2.INTER_LINEAR)
        img_array = (img_array-np.min(img_array))/(np.max(img_array)-np.min(img_array))
        img_show[n,:] = img_array
        n+=1

plt.imshow(img_show[8])
plt.show()

n = 0
#img = Image.open('./selfie2anime/testA/female_10328.jpg')
#dim = (255,255,3)
path = './selfie2anime/testA'
img_h = np.zeros([100,256,256,3])
for path,dir,file in os.walk(path):
    for img_get in file:
        img_open = Image.open(path+'/'+img_get)
        #plt.imshow(img_open)
        img_array = np.array(img_open)
        #img_resize = cv2.resize(img_array,dim,interpolation = cv2.INTER_LINEAR)
        img_array = (img_array-np.min(img_array))/(np.max(img_array)-np.min(img_array))
        img_h[n,:] = img_array
        n+=1
        
plt.imshow(img_h[3])
plt.show()

img_data = np.vstack((img_show, img_h))
y = np.zeros((200))
y[100:]=1
print(img_data.shape)
print(y[5])


# In[561]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(img_data, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(y_train.shape)
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)


# In[562]:


from keras.applications.resnet50 import ResNet50
from keras.applications import MobileNetV2
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Input
from keras import backend as K
from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
from torchvision import models, transforms
import torch.nn.functional as F
import numpy as np
import skimage.transform

#conv_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential()

model.add(Conv2D(12, kernel_size=(3, 3), activation='relu', padding='same',input_shape=(256, 256,3)))
# model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(MyCustomLayer(32,use_bias = False))
'''model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
# # model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same'))
# # model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
'''

# model.add(conv_base)
model.add(Flatten())
# model.add(Dense(64, activation = 'relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='Adam',
              metrics=['accuracy'])
#kernel_regularizer=l2(1e-2)
# model.fit(x_train, y_train,
#           batch_size=64,
#           epochs=10,
#           verbose=1,
#           validation_data=(x_test, y_test))
history = model.fit(
        X_train,y_train,
        epochs=20,
        validation_data=(X_test, y_test))


# In[567]:


img = np.reshape(img_show[35],(1,256,256,3))
iimage = Image.open("./selfie2anime/testB/0000.jpg")
plt.imshow(img_show[35])
model.predict(img)


# In[568]:


model.summary()


# In[569]:


from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Two class recognition')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[570]:


tf.keras.layers.Layer.trainable_weights


# In[573]:


def getCAM(feature_conv, weight_fc, class_idx):
    _, h, w, nc = feature_conv.shape
    cam = weight_fc[class_idx]*(feature_conv.reshape((h * w, nc)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

layer_1 = K.function([model.layers[0].input], [model.get_layer('max_pooling2d_51').output])#第一个 model.layers[0],不修改,表示输入数据；第二个model.layers[you wanted],修改为你需要输出的层数的编号
f1 = layer_1([img])[0]#只修改inpu_image
#第一层卷积后的特征图展示，输出是（1,149,149,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
print(f1.shape)
X = f1.flatten()


output = model.get_layer('flatten_24')(f1)
#weight_softmax_params = model.get_layer('flatten_21').get_weights()
layer_flat = K.function([model.layers[0].input], [model.get_layer('flatten_24').output])
f2 = layer_flat([img])[0]
print(f2.shape)
#features = model.features(X)
#layer_2 = K.function([model.layers[0].input], [model.get_layer('max_pooling2d_26').output])
preds = model.predict(img)
print(preds)

#weight_softmax = np.squeeze(weight_softmax_params[0].data.numpy())
fig, ax = plt.subplots(3, 4, sharex='col', sharey='row')

n=0
for i in range(3):
    for j in range(4):
            show_img = f1[:, :, :, n]
            show_img.shape = [128, 128]
            #plt.figure(figsize=(5,5))
            #plt.subplot(3, 3, _ + 1)
            ax[i,j].imshow(show_img, cmap='gray')
            #plt.axis('off')
            n+=1
            

weight_softmax = np.squeeze(f2[0])

prediction = model(img)
#pred_probabilities = F.softmax(prediction).data.squeeze()
#class_idx = torch.topk(pred_probabilities, 1)[1].int()
print(class_idx)



# In[578]:


# Get the features from a model
class SaveFeatures():
    features = None
    def __init__(self, module): 
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): 
        self.features = output.data.numpy()

    def remove(self): 
        self.hook.remove()

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h * w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((256, 256)),
   transforms.ToTensor(),
   normalize,
])
x_img = preprocess(iimage).unsqueeze(0)

print(x_img.shape)
model = models.resnet18(pretrained=True)


# In[579]:





final_layer = model._modules.get('layer4')
activated_features = SaveFeatures(final_layer)

# Inference
model.eval()
print(img.shape)
prediction = model(x_img)
pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove()
# Take weights from the first linear layer
weight_softmax_params = list(model._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].data.numpy())

# Get the top-1 prediction and get CAM
class_idx = torch.topk(pred_probabilities, 1)[1].int()
print(class_idx)
overlay = getCAM(activated_features.features, weight_softmax, class_idx )


# In[580]:



plt.figure(figsize=(5, 5))
plt.title('Class Activation Map', fontweight='bold')
plt.imshow(overlay[0], alpha=0.5, cmap='jet')

# Show CAM on the image
plt.figure(figsize=(15, 10))
plt.title('CAM Image')
plt.imshow(iimage)
plt.imshow(skimage.transform.resize(overlay[0], (iimage.size[1], iimage.size[0])), alpha=0.5, cmap='jet');
plt.show()


# In[ ]:




