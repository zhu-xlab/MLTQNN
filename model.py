import tensorflow as tf
import tensorflow_quantum as tfq
import sympy

import cirq
from cirq.contrib.svg import SVGCircuit
from cirq.circuits.qasm_output import QasmUGate
from tensorflow_quantum.python.layers.circuit_executors import expectation, sampled_expectation
from tensorflow_quantum.python.layers.circuit_construction import elementary
from tensorflow_quantum.python import util

import numpy as np
import random
import os
import struct
from array import array
from os.path  import join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL import Image, ImageFilter
import scipy.io

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from datetime import datetime
import yaml
from pathlib import Path

import warnings
import tensorflow.keras.backend as K
import json

import patchEncoder
from dataLoader import DataLoader
import callbacks
import pqcCNN

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, kernels, name="extraction"):
        super(FeatureExtractor, self).__init__(name=name)
        
        layers = []
        for kernel in kernels:
            layers.append(tf.keras.layers.Conv2D(filters=kernel, kernel_size=2, strides=(1, 1), padding='same'))
            layers.append(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
            layers.append(tf.keras.layers.ReLU())
        self.units = tf.keras.Sequential(layers)
        self.flatten = tf.keras.layers.Flatten()
        
            
    def call(self, inputs):
        features = self.units(inputs)
        features = self.flatten(features)
        return features


def get_qcnn_model(dataset, path, locWeight, input_shape, categories):
    if dataset == 'sat' or dataset == 'lcz':
        encoder_filters = [12]
        decoder_filters = [12]
        patch_size = 4
    if dataset == 'eurosat':
        encoder_filters = [12, 16]
        decoder_filters = [16, 12]
        patch_size = 8
    if dataset == 'patternet':
        encoder_filters = [12,16,20,24]  
        decoder_filters = [24,20,16,12]
        patch_size = 32        
    inputs = tf.keras.Input(shape=input_shape, name='input')
    patches = patchEncoder.Patches(patch_size, input_shape, name='patch')(inputs)
    encoder = patchEncoder.Encoder(encoder_filters, name='encoder')(patches)
    decoder = patchEncoder.Decoder(decoder_filters, input_shape[-1], name='decoder')(encoder)
    reconstruct = patchEncoder.Unpatch((patch_size, patch_size, input_shape[-1]), input_shape, name='reconstruction')(decoder)
    
    if locWeight:
        qpc_layer = pqcCNN.allEntEncodingPQC_u3loc(name='pqc')(encoder)
    if not locWeight:
        qpc_layer = pqcCNN.allEntEncodingPQC_woLoc(name='pqc')(encoder)
    prediction = tf.keras.layers.Dense(categories, activation='softmax', name='classifier')(qpc_layer)
    qcnn_model = tf.keras.Model(inputs=[inputs], outputs=[prediction, reconstruct])
    # print(qcnn_model.summary())    
    
    
    batch_size = 50
    lr = 0.01

    alpha = K.variable(1.0)
    beta = K.variable(1.0)

    losses = {"classifier": "categorical_crossentropy", "reconstruction": "mse"}
    lossWeights = {"classifier": alpha, "reconstruction": beta}
    metrics = {'classifier': 'accuracy'}


    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                       loss=losses, loss_weights=lossWeights, metrics=metrics)
    
    qcnn_model.load_weights(path)
    return qcnn_model


def get_cnn_based_model(dataset, path, input_shape, categories):
    if dataset == 'sat' or dataset == 'lcz':
        encoder_filters = [12]
        decoder_filters = [12]
        patch_size = 4
    if dataset == 'eurosat':
        encoder_filters = [12, 16]
        decoder_filters = [16, 12]
        patch_size = 8
    if dataset == 'patternet':
        encoder_filters = [12,16,20,24]  
        decoder_filters = [24,20,16,12]
        patch_size = 32        
    inputs = tf.keras.Input(shape=input_shape, name='input')
    patches = patchEncoder.Patches(patch_size, input_shape, name='patch')(inputs)
    encoder = patchEncoder.Encoder(encoder_filters, name='encoder')(patches)
    decoder = patchEncoder.Decoder(decoder_filters, input_shape[-1], name='decoder')(encoder)
    reconstruct = patchEncoder.Unpatch((patch_size, patch_size, input_shape[-1]), input_shape, name='reconstruction')(decoder)
    
    qpc_layer = FeatureExtractor([2, 16], name='pqc')(encoder)

    prediction = tf.keras.layers.Dense(categories, activation='softmax', name='classifier')(qpc_layer)
    qcnn_model = tf.keras.Model(inputs=[inputs], outputs=[prediction, reconstruct])
    # print(qcnn_model.summary())    
    
    
    batch_size = 50
    lr = 0.01

    alpha = K.variable(1.0)
    beta = K.variable(1.0)

    losses = {"classifier": "categorical_crossentropy", "reconstruction": "mse"}
    lossWeights = {"classifier": alpha, "reconstruction": beta}
    metrics = {'classifier': 'accuracy'}


    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                       loss=losses, loss_weights=lossWeights, metrics=metrics)
    
    qcnn_model.load_weights(path)
    return qcnn_model



def configofmodel(id_str, model_type, wandb):
    files = os.listdir(wandb)
    target = ''
    for file in files:
        if id_str in file:
            path = os.path.join(wandb, file + '/files/config.yaml')
            return path   

            
            
def jsonofmodel(id_str, model_type, wandb):
    files = os.listdir(wandb)
    target = ''
    for file in files:
        if id_str in file:
            path = os.path.join(wandb , file + '/files/wandb-summary.json')
            with open(path, 'r') as file:
                data = file.read()
                parsed_data = json.loads(data)

            return parsed_data['test_best_acc']
            
        
def pathofmodel(id_str, model_type, wandb):
    files = os.listdir(wandb)
    target = ''
    for file in files:
        if id_str in file:
            path = os.path.join(wandb, file + '/files/model-best.h5')
            return path   
    
def get_model_config(target_run, name, model_type, wandb):
    run_id  = target_run.split('/')[-1]
    path = configofmodel(run_id, model_type, wandb)
    config = yaml.safe_load(Path(path).read_text())
    
    label_ratio = config['label_ratio']['value']
    
    run_meta = run_id.split('_')
    dataset = run_meta[0]
    
    if 'U3locWeight' in run_meta:
        locWeight = True
    else:
        locWeight = False
        
    
    path = pathofmodel(run_id, model_type, wandb)

    name = [name]
    if locWeight:
        name.append('U3locWeight')

    model_name = '_'.join(name)    

    
    # print(dataset)
    # print('ratio', label_ratio)
    # print('locWeight', locWeight)
    # print(model_name)
    
    test_acc = jsonofmodel(run_id, model_type, wandb)

    return label_ratio, dataset, locWeight, model_name, path, test_acc


def matrices(model, train_x, train_y, target):
    results = model.evaluate(train_x, [train_y, train_x])
    return results
    


def build_model(model_type, dataset, label_ratio, input_shape, categories, target_run, wandb):
    if model_type == 'qcnn':
        # print(target_run)
        obtained_label_ratio, obtained_dataset, locWeight, model_name, path, test_acc = get_model_config(target_run, 'QCNN_patch', model_type, wandb)
        assert obtained_label_ratio == label_ratio
        assert obtained_dataset == dataset
        
        qcnn_model = get_qcnn_model(dataset, path, locWeight, input_shape, categories)
        

        return qcnn_model, test_acc

    
    if model_type == 'cnn_based':
        obtained_label_ratio, obtained_dataset, locWeight, model_name, path, test_acc = get_model_config(target_run, 'EncoderCNN', model_type, wandb)
        assert obtained_label_ratio == label_ratio
        assert obtained_dataset == dataset    
        qcnn_model = get_cnn_based_model(dataset, path, input_shape, categories)
    
        return qcnn_model, test_acc

