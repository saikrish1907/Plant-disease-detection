#from google.colab import drive

# This will prompt for authorization.
#drive.mount('/content/drive')


import numpy as np
np.random.seed(1337)
from tensorflow import set_random_seed
set_random_seed(1923)

import pickle
import cv2
import os
import glob
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model


EPOCHS = 20
INIT_LR = 1e-3
BS = 32
default_image_size = tuple((256, 256))
image_size = 0
directory_root = 'D:/Plant disease detection/train'
width=256
height=256
depth=3




"""def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print("Error : {e}")
        return None
"""


image_list, label_list = [], []
bg_list= []
root_dir=os.listdir(directory_root)


for directory in root_dir :
        # remove .DS_Store from list
        if directory == ".DS_Store" :
            root_dir.remove(directory)

for image_dir in root_dir:
	    image_directory=f"{root_dir}/{image_dir}"

#label_list.append(image_directory)

##Taking data from each folder 
for img in glob.glob("D:/Plant disease detection/train/Pepper__bell___Bacterial_spot/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Pepper__bell___Bacterial_spot")

for img in glob.glob("D:/Plant disease detection/train/Pepper__bell___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Pepper__bell___healthy")

for img in glob.glob("D:/Plant disease detection/train/Potato___Early_blight/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Potato___Early_blight")

for img in glob.glob("D:/Plant disease detection/train/Potato___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Potato___healthy")

for img in glob.glob("D:/Plant disease detection/train/Potato___Late_blight/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Potato___Late_blight")

for img in glob.glob("D:/Plant disease detection/train/Tomato___Target_Spot/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato__Target_Spot")

for img in glob.glob("D:/Plant disease detection/train/Tomato___Tomato_mosaic_virus/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato__Tomato_mosaic_virus")

for img in glob.glob("D:/Plant disease detection/train/Tomato___Tomato_Yellow_Leaf_Curl_Virus/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato__Tomato_YellowLeaf__Curl_Virus")

for img in glob.glob("D:/Plant disease detection/train/Tomato___Bacterial_spot/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato_Bacterial_spot")

for img in glob.glob("D:/Plant disease detection/train/Tomato___Early_blight/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato_Early_blight")

for img in glob.glob("D:/Plant disease detection/train/Tomato___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato_healthy")

for img in glob.glob("D:/Plant disease detection/train/Tomato___Late_blight/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato_Late_blight")

for img in glob.glob("D:/Plant disease detection/train/Tomato___Leaf_Mold/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato_Leaf_Mold")

for img in glob.glob("D:/Plant disease detection/train/Tomato___Septoria_leaf_spot/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato_Septoria_leaf_spot")

for img in glob.glob("D:/Plant disease detection/train/Tomato___Spider_mites Two-spotted_spider_mite/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Tomato_Spider_mites_Two_spotted_spider_mite")



for img in glob.glob("D:/Plant disease detection/train/Apple___Apple_scab/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Apple___Apple_scab")

for img in glob.glob("D:/Plant disease detection/train/Apple___Black_rot/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Apple___Black_rot")

for img in glob.glob("D:/Plant disease detection/train/Apple___Cedar_apple_rust/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Apple___Cedar_apple_rust")


for img in glob.glob("D:/Plant disease detection/train/Apple___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Apple___healthy")


for img in glob.glob("D:/Plant disease detection/train/Blueberry___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Blueberry___healthy")

for img in glob.glob("D:/Plant disease detection/train/Cherry_(including_sour)___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Cherry_(including_sour)___healthy")	

for img in glob.glob("D:/Plant disease detection/train/Cherry_(including_sour)___Powdery_mildew/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Cherry_(including_sour)___Powdery_mildew")			

for img in glob.glob("D:/Plant disease detection/train/Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot")			

for img in glob.glob("D:/Plant disease detection/train/Corn_(maize)___Common_rust_/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Corn_(maize)___Common_rust_")			

for img in glob.glob("D:/Plant disease detection/train/Corn_(maize)___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Corn_(maize)___healthy")			

for img in glob.glob("D:/Plant disease detection/train/Corn_(maize)___Northern_Leaf_Blight/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Corn_(maize)___Northern_Leaf_Blight")			

for img in glob.glob("D:/Plant disease detection/train/Grape___Black_rot/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Grape___Black_rot")			

for img in glob.glob("D:/Plant disease detection/train/Grape___Esca_(Black_Measles)/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Grape___Esca_(Black_Measles)")			

for img in glob.glob("D:/Plant disease detection/train/Grape___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Grape___healthy")			

for img in glob.glob("D:/Plant disease detection/train/Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)")			

for img in glob.glob("D:/Plant disease detection/train/Orange___Haunglongbing_(Citrus_greening)/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Orange___Haunglongbing_(Citrus_greening)")			

for img in glob.glob("D:/Plant disease detection/train/Peach___Bacterial_spot/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Peach___Bacterial_spot")			

for img in glob.glob("D:/Plant disease detection/train/Peach___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Peach___healthy")			

for img in glob.glob("D:/Plant disease detection/train/Raspberry___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Raspberry___healthy")			

for img in glob.glob("D:/Plant disease detection/train/Soybean___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Soybean___healthy")			

for img in glob.glob("D:/Plant disease detection/train/Squash___Powdery_mildew/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Squash___Powdery_mildew")			

for img in glob.glob("D:/Plant disease detection/train/Strawberry___healthy/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Strawberry___healthy")			

for img in glob.glob("D:/Plant disease detection/train/Strawberry___Leaf_scorch/*.JPG"):
		img=cv2.imread(img,1)
		image_list.append(img)
		label_list.append("Strawberry___Leaf_scorch")			

"""for img in glob.glob("D:/Plant disease detection/train/images/*.jpg"):
		img=cv2.imread(img,1)
		bg_list.append(img)
		label_list.append("bgimages")			

		#print(image_list)    
"""

#bg_list=convert_image_to_array(bg_list)
#img_len=len(bg_list)
#print(img_len)
image_size = len(image_list)
print(image_size)
#print(label_list)
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)  

print(label_binarizer.classes_)
np_image_list = (255 - np.array(image_list, dtype=np.float16) ) / 225.0
#bg_image_list = (255 - np.array(bg_list, dtype=np.float16) ) / 225.0
#bg_image_list=(255-bg_list)/225.0
#print(np_image_list)
#print(bg_image_list)
#np_image_list=np_image_list+bg_image_list
img_len=len(np_image_list)
print(img_len)
#Training of data
print("Splitting dataset to training and test data")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.25, random_state = 42)  
  

#
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")



model=Sequential()
inputShape = (height, width, depth)

chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1


model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))


model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))


model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
#model.add(Conv2D(64, (3, 3), padding="same"))
#model.add(Activation("relu"))
#model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))



model.summary()

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print(" Training network...")

history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=BS),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, verbose=1 ,shuffle=False
    )
    

print("Calculating our model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")  
#model.save('cnn_file.h5')  

#pickle.dump(model,open('cnn_model.pkl', 'wb'))
model.save('my_model.h5')

print("Saved the model successfully")