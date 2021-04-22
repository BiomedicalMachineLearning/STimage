import os; import sys; import pandas as pd; import numpy as np; from numpy import array; from numpy import argmax

from keras.applications.imagenet_utils import decode_predictions
from keras.utils import to_categorical; from sklearn.preprocessing import LabelEncoder; from sklearn.preprocessing import OneHotEncoder; from sklearn.model_selection import train_test_split

import tensorflow as tf; from matplotlib import pyplot as plt
from keras import backend; from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential; from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD; from keras.models import Model; from tensorflow.keras import regularizers; from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input as pi; from keras.callbacks import ModelCheckpoint

import tensorflow.keras.backend as K; from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions; import cv2

import lime; from lime import lime_image
from keras.preprocessing import image; import numpy as np; import h5py; from keras.models import load_model 

#%%

def train_interpretation_model():
    '''
    Gene_exp_train = pd.read_csv('pivot_trial_Y.csv').sort_values(by="Sno").reset_index()
    Biomarker_train = Gene_exp_train[['MALAT1']]
    Biomarker_train = pd.DataFrame(data=Biomarker_train)
    Biomarker_train = Biomarker_train.apply(lambda x: pd.qcut(x, 3,duplicates='drop', labels=False))
    Biomarker_train["Image"] = Gene_exp_train[['Sno']] + str(".tif")
    Biomarker_train = Biomarker_train.rename(columns={Biomarker_train.columns[0]: 'Gene_cluster'})
    Biomarker_train["Gene_cluster"] = Biomarker_train["Gene_cluster"].astype('str') 
    
    Gene_exp_test = pd.read_csv('pivot_test_Y.csv').sort_values(by="Sno").reset_index()
    Biomarker_test = Gene_exp_test[['MALAT1']]
    Biomarker_test = pd.DataFrame(data=Biomarker_test)
    Biomarker_test = Biomarker_test.apply(lambda x: pd.qcut(x, 3,duplicates='drop', labels=False))
    Biomarker_test["Image"] = Gene_exp_test[['Sno']] + str(".tif")
    Biomarker_test = Biomarker_test.rename(columns={Biomarker_test.columns[0]: 'Gene_cluster'})
    Biomarker_test["Gene_cluster"] = Biomarker_test["Gene_cluster"].astype('str')
    datagen=ImageDataGenerator(rescale=1./255, validation_split = 0.2, featurewise_center=True,
            featurewise_std_normalization=False,rotation_range=90,
            width_shift_range=0.2,height_shift_range=0.2,
            horizontal_flip=True,vertical_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255,featurewise_center=True)
    
    train_generator=datagen.flow_from_dataframe(dataframe=Biomarker_train, directory="Trainimg_breast_2_299", 
                                                x_col="Image", y_col="Gene_cluster", class_mode="categorical", 
                                                target_size=(290,290), batch_size=32, subset="training")
    valid_generator = datagen.flow_from_dataframe(dataframe=Biomarker_train, directory="Trainimg_breast_2_299", 
                                                  x_col="Image", y_col="Gene_cluster", class_mode="categorical", 
                                                  target_size=(290,290), batch_size=32, subset="validation")
    test_generator = test_datagen.flow_from_dataframe(dataframe=Biomarker_test, directory="Trainimg_breast_test_299", 
                                                      x_col="Image", y_col="Gene_cluster", class_mode="categorical", 
                                                      target_size=(290,290), batch_size=32)
    model = ResNet50(include_top=False, input_shape=(290,290,3), weights = "imagenet")
    flat1 = Flatten()(model.layers[-1].output)
    dense = Dense(256, activation='relu')(flat1)
    drop = Dropout(0.5)(dense)
    output = Dense(3, activation='softmax')(drop)
    model = Model(inputs=model.inputs, outputs=output)
    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
                             save_best_only=True, mode='auto', period=1)


    model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['AUC'])
    history = model.fit(train_generator,validation_data=valid_generator,epochs=10, 
                        callbacks=[checkpoint])
    '''
    
    model = load_model("Inception_model_COX6C_18") 
    return model


def LIME_heatmaps(im, sav):
    
    def transform_img_fn(path_list):
        out = []
        for img_path in path_list:
            img = image.load_img(img_path, target_size=(290, 290))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = pi(x)
            out.append(x)
        return np.vstack(out)
    
    pred_val1 = []; pred_val2 = []; pred_val3 = [] 
    for i in range(0,len(im)):
        images = transform_img_fn([os.path.join('Trainimg_breast_2_299',im[i])])
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(images[0].astype('double'), train_interpretation_model().predict, top_labels=3, num_samples=8)
        dict_heatmap1 = dict(explanation.local_exp[explanation.top_labels[0]])
        dict_heatmap2 = dict(explanation.local_exp[explanation.top_labels[1]])
        dict_heatmap3 = dict(explanation.local_exp[explanation.top_labels[2]])
    
        pred_val1.append(explanation.top_labels[0])
        pred_val2.append(explanation.top_labels[1])
        pred_val3.append(explanation.top_labels[2])
    
        heatmap1 = np.vectorize(dict_heatmap1.get)(explanation.segments)
        heatmap1 = np.maximum(heatmap1, 0)
        heatmap1 /= np.max(heatmap1)
    
        heatmap2 = np.vectorize(dict_heatmap2.get)(explanation.segments) 
        heatmap2 = np.maximum(heatmap2, 0)
        heatmap2 /= np.max(heatmap2)
        
        heatmap3 = np.vectorize(dict_heatmap3.get)(explanation.segments) 
        heatmap3 = np.maximum(heatmap3, 0)
        heatmap3 /= np.max(heatmap3)
    
        Lime1 = plt.imsave(str(explanation.top_labels[0])+"_"+sav[i], heatmap1, cmap = 'RdBu', vmin  = -heatmap1.max(), vmax = heatmap1.max())
        Lime2 = plt.imsave(str(explanation.top_labels[1])+"_"+sav[i], heatmap2, cmap = 'RdBu', vmin  = -heatmap2.max(), vmax = heatmap2.max())
        Lime3 = plt.imsave(str(explanation.top_labels[2])+"_"+sav[i], heatmap3, cmap = 'RdBu', vmin  = -heatmap3.max(), vmax = heatmap3.max())
        
        return Lime1, Lime2, Lime3

#im = ['0073img.tif']; sav = ['0073.png']
#LIME_heatmaps(im,sav)

def activation_maps():
    
    model = load_model("Inception_model_COX6C_18") 
    
    img = image.load_img('Trainimg_breast_2_299/0073img.tif', target_size=(290,290))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #print(decode_predictions(predicted_vals[0,:]))
    
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv5_block3_2_relu')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((10,10))
    act_map = plt.matshow(heatmap)
    plt.show()
    return act_map

#activation_maps()

#%%