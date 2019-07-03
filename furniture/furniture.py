import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import flags
from datetime import datetime
import sys
import os

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.xception import Xception

from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.backend import set_session

from tensorflow.python.keras.layers import Input, Flatten, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.utils import plot_model

from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D

sys.path.append("/home/jhyun/keras/")

import get_train_data

def set_tf_session():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.Session(config=config)
    set_session(sess)
    print("Global session configured")

def process_images_thru_tl(batch_size=32, input1=1024, input2=1024, model_name="vgg"):

    with tf.device("/device:GPU:1"):
        if model_name == "vgg":
            model = VGG16(weights = "imagenet", include_top=False, input_shape = [224, 224, 3])
            size = 224
        elif model_name == "inceptionv3":
            model = InceptionV3(weights = "imagenet", include_top=False, input_shape = [299, 299, 3])
            size = 299
        elif model_name == "resnet50":
            model = ResNet50(weights = "imagenet", include_top=False, input_shape = [224, 224, 3])
            size = 224
        elif model_name == "mobilenet":
            model = ResNet50(weights = "imagenet", include_top=False, input_shape = [224, 224, 3])
            size = 224
        elif model_name == "xception":
            model = Xception(weights = "imagenet", include_top=False)
            size = 299


    print("%s %d %d %d" % (model_name, input1, input2, batch_size))
    model.summary()
    model.get_weights()
  
    labels = []
    batch = []

#    input_ =  Input(shape=(size,size,3),name = 'image_input')
    output_ = model.output

    with tf.device("/device:GPU:1"):
        if model_name == "inceptionv3" or  model_name == "xception":
            x = GlobalAveragePooling2D(name='avg_pool')(output_)
        else:
            x = Flatten(name='flatten')(output_)
        if input1 != 0:
            x = Dense(input1, activation='relu', name='fc1')(x)
        if input2 != 0:
            x = Dense(input2, activation='relu', name='fc2')(x)

        x = Dense(128, activation='softmax', name='predictions')(x)
    for layer in model.layers:
        layer.trainable = False
    
    my_model = Model(inputs=output_, outputs=x)
    my_model.summary()

    if os.path.exists("weights_%s_%d_%d_%d.h5" % (model_name, input1, input2, batch_size)):
        my_model.load_weights("weights_%s_%d_%d_%d.h5" % (model_name, input1, input2, batch_size))


    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    train_generator = get_train_data.train_generator(img_size=size, batch_size=batch_size)
    valid_generator = get_train_data.valid_generator(img_size=size, batch_size=8)
    csv_logger = CSVLogger('log.csv', append=True, separator=',')
    my_model.fit_generator(
            train_generator,
            steps_per_epoch=2000,#1000
            epochs=10,
            validation_data=valid_generator,
            validation_steps=200,#200
            callbacks=[csv_logger])
    my_model.save_weights("weights_%s_%d_%d_%d.h5" % (model_name, input1, input2, batch_size))


if __name__ == "__main__":
    set_tf_session()
    if len(sys.argv) != 4:
        print("Usage: model, fc1_size, fc2_size")
        exit()
    else:
        model= sys.argv[1]
        fc1_size = int(sys.argv[2])
        fc2_size = int(sys.argv[3])
        process_images_thru_tl(batch_size=8, input1=fc1_size, input2=fc2_size, model_name=model)




