

import numpy as np
import pickle
import cv2
import keract
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
from sklearn.preprocessing import MultiLabelBinarizer

from keras.models import load_model

img_li=[] 
width=256
height=256
depth=3
test_model = load_model('D:/Study material/Plant disease/Model_final/Regulariser/my_model.h5')
#test_model = pickle.load(open('/content/cnn_model.pkl', 'rb'))
#test_model = load_model('D:/Study material/Plant disease/mod_all/my_model_97.h5')
print("Hello")
default_image_size = tuple((256, 256))


#img = cv2.imread('/content/drive/My Drive/straw1.jpg',1)
img = cv2.imread('D:/xampp/htdocs/application/img/IMG_TEST.JPG',1)
if img is not None :
                                    img = cv2.resize(img, default_image_size)   
                                    img_li=img_to_array(img)
else :
                                    img_li=np.asarray(img)

#img_li=img_to_array(image)

np_image_list = (255-np.array(img_li, dtype=np.float16)) / 225.0
#np_image_list = (np.array(img_li, dtype=np.float16)) / 225.0
inputShape = (height, width, depth)

aug = ImageDataGenerator(
            rotation_range=25, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2, 
            zoom_range=0.2,horizontal_flip=True, 
            fill_mode="nearest")

chanDim = -1
if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1



#np_image_list = np.random.randint(0,10,inputShape)
np_image_list= np.expand_dims(np_image_list, axis=0)

ypredict=test_model.predict_classes(np_image_list)

print(ypredict)

"""activations = keract.get_activations(test_model, np_image_list)
first = activations.get('block1_conv1/Relu:0')
keract.display_activations(activations)
"""

if ypredict==0:
   result="The give leaf is apple and has a disease called as Apple___Apple_scab"
elif ypredict==1:
   result="The give leaf is apple and has a disease called as Apple___Black_rot"
elif ypredict==2:
   result="The give leaf is apple and has a disease called as Apple___Cedar_apple_rust"
elif ypredict==3:
   result="The give leaf is apple and is  healthy"
elif ypredict==4:
   result="The give leaf is blueberry and has no disease (Blueberry___healthy)"
elif ypredict==5:
   result="The give leaf is cherry and has a disease called as Cherry_(including_sour)___Powdery_mildew"
elif ypredict==6:
   result="The give leaf is cherry and has no disease "
elif ypredict==7:
   result="The give leaf is corn and has a disease called as Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
elif ypredict==8:
   result="The give leaf is corn and has a disease called as Corn_(maize)___Common_rust_"
elif ypredict==9:
   result="The give leaf is corn and has a disease called as Corn_(maize)___Northern_Leaf_Blight"
elif ypredict==10:
   result="The give leaf is corn and has no disease"
elif ypredict==11:
   result="The give leaf is grape and has a disease called as Grape___Black_rot"
elif ypredict==12:
   result="The give leaf is grape and has a disease called as Grape___Esca_(Black_Measles)"
elif ypredict==13:
   result="The give leaf is grape and has a disease called as Grape___Leaf_blight_(Isariopsis_Leaf_Spot)"
elif ypredict==14:
   result="The give leaf is grape and has no disease"
elif ypredict==15:
   result="The give leaf is orange and has a disease called Orange___Haunglongbing_(Citrus_greening)"
elif ypredict==16:
   result="The give leaf is peach and has a disease called Peach___Bacterial_spot"
elif ypredict==17:
   result="The give leaf is peach and has no disease"
elif ypredict==18:
   result="The give leaf is Potato and has a disease called as Potato___Early_blight"
elif ypredict==19:
   result="The give leaf is Potato and has a disease called as Potato___Late_blight"
elif ypredict==20:
   result="The give leaf is Potato and has no disease "
elif ypredict==21:
   result="The give leaf is Raspberry and has no disease"
elif ypredict==22:
   result="The give leaf is Squash and has a disease called Squash___Powdery_mildew"
elif ypredict==23:
   result="The give leaf is Strawberry and has disease called Strawberry___Leaf_scorch"
elif ypredict==24:
   result="The give leaf is Strawberry and has no  disease"
elif ypredict==25:  
   result="The give leaf is Tomato and has a disease called as Tomato_Bacterial_spot"
elif ypredict==26:
   result="The give leaf is Tomato and has a disease called as Tomato_Early_blight" 
elif ypredict==27:
   result="The give leaf is Tomato and has a disease called as Tomato_Late_blight"
elif ypredict==28:
   result="The give leaf is Tomato and has a disease called as Tomato_Leaf_mold"
elif ypredict==29:
   result="The give leaf is Tomato and has a disease called as Tomato_Septoria_leaf_spot"
elif ypredict==30:
   result="The give leaf is Tomato and has a disease called as Tomato_Spider_mites"
elif ypredict==31:
   result="The give leaf is Tomato and has a disease called as Tomato_target_spot"
elif ypredict==32:
   result="The give leaf is Tomato and has a disease called as Tomato_yellow_leaf_curl_virus"
elif ypredict==33:
   result="The give leaf is Tomato and has a disease called as Tomato_mosaic_virus"
elif ypredict==34:
   result="The give leaf is Tomato and has no disease "
else:
   result="Trained as background image for the given model"

print(result)