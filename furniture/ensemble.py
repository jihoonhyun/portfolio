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

from tensorflow.python.keras._impl.keras.layers import GlobalAveragePooling2D, Average

from tensorflow.python.keras.models import load_model

sys.path.append("/home/jhyun/keras/")

import get_train_data

def predict_with_pretrained_model(valid_generator, name, fc1, fc2, batch_size):
    model = load_model("model_%s_%d_%d_%d.h5" % (name, fc1, fc2, batch_size))
    model.load_weights("weights_%s_%d_%d_%d.h5" % (name, fc1, fc2, batch_size))
    nb_samples = len(valid_generator.filenames)
    #nb_samples = 200
    predict = model.predict_generator(valid_generator, steps=nb_samples, verbose = 1)
    #predict = model.evaluate_generator(valid_generator, steps=nb_samples)
    #print predict
    return predict

def evaluate_single_model(name, fc1, fc2, batch_size, size):
    valid_generator = get_train_data.valid_generator(img_size=size, batch_size=8)
    model = load_model("model_%s_%d_%d_%d.h5" % (name, fc1, fc2, batch_size))
    model.load_weights("weights_%s_%d_%d_%d.h5" % (name, fc1, fc2, batch_size))
#    nb_samples = len(valid_generator.filenames)
    nb_samples = 200
    predict = model.evaluate_generator(valid_generator, steps=nb_samples)
    print("%s,%d,%d,%d, %f, %f" %  (name, fc1, fc2, batch_size, predict[0], predict[1]))

def evaluate(valid_generator, predict):
    result = 0.0
    label_predicted = []
    for batch in predict:
        label_predicted.append(np.argmax(batch))

    for i in range(len(label_predicted)):
        x, y = valid_generator.next()
        label = np.argmax(y)
        if label_predicted[i] == label:
            result += 1
    return result / len(label_predicted)

def geo_mean(predict):
    num = 5
    result = []
    for i in range(len(predict[0])):
        res_list = []
        for j in range(128):
            res = 1
            for z in range(num):
                res = res * predict[z][i][j]
            res = np.power(res, 1.0/num)
            res_list.append(res)
        result.append(res_list)
    return result

if __name__ == "__main__":
    predict_raw = []
    valid_generator = get_train_data.predict_generator(img_size=299, batch_size=8)
    predict = predict_with_pretrained_model( valid_generator, "inceptionv3", 0, 0, 8)
    predict_raw.append(predict)

    valid_generator = get_train_data.predict_generator(img_size=299, batch_size=8)
    predict = predict_with_pretrained_model( valid_generator, "xception", 0, 0, 8)
    predict_raw.append(predict)


    valid_generator = get_train_data.predict_generator(img_size=224, batch_size=8)
    predict = predict_with_pretrained_model( valid_generator, "resnet50", 1024, 1024, 8)
    predict_raw.append(predict)

    valid_generator = get_train_data.predict_generator(img_size=224, batch_size=8)
    predict = predict_with_pretrained_model( valid_generator, "vgg", 1024, 1024, 8)
    predict_raw.append(predict)

    valid_generator = get_train_data.predict_generator(img_size=224, batch_size=8)
    predict = predict_with_pretrained_model( valid_generator, "mobilenet", 1024, 1024, 8)
    predict_raw.append(predict)
 
    predict = geo_mean(predict_raw)
    valid_generator = get_train_data.predict_generator(img_size=299, batch_size=1)
    result = evaluate(valid_generator, predict)
    print result 
#    evaluate_single_model("inceptionv3", 0, 0, 8, 299)
#    evaluate_single_model("xception", 0, 0, 8, 299)
#    evaluate_single_model("resnet50", 1024, 1024, 8, 224)
#    evaluate_single_model("vgg", 1024, 1024, 8, 224)
#    evaluate_single_model("mobilenet", 1024, 1024, 8, 224)

#    print predict_inv3
#    print predict_xcep
    #print predict

    #valid_generator = get_train_data.predict_generator(img_size=224, batch_size=8)






