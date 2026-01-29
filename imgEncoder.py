import tensorflow as tf
import numpy as np


# class Encoder(tf.keras.layers.Layer):
#     def __init__(self, kernels, name="encoder"):
#         super(Encoder, self).__init__(name=name)
#         nElements = 9 
        
#         layers = []
#         for kernel in kernels:
#             layers.append(tf.keras.layers.Conv2D(filters=kernel, kernel_size=2, strides=(1, 1), padding='same'))
#             layers.append(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
#             layers.append(tf.keras.layers.BatchNormalization())
#             layers.append(tf.keras.layers.ReLU())
        
#         layers.append(tf.keras.layers.Conv2D(filters=nElements, kernel_size=1, strides=(1, 1), padding='same'))
#         self.units = tf.keras.Sequential(layers)
            
#     def call(self, inputs):
#         encoded = self.units(inputs)
#         return encoded

# class Reconstruction(tf.keras.layers.Layer):
#     def __init__(self, kernels, nchannel, name="reconstruction"):
#         super(Reconstruction, self).__init__(name=name)
        
#         layers = []
#         layers.append(tf.keras.layers.Conv2D(filters=kernels[0], kernel_size=1, strides=(1, 1), padding='same'))

#         for kernel in kernels[1:]:
#             layers.append(tf.keras.layers.Conv2DTranspose(filters=kernel, kernel_size=2, strides=2, padding='same'))
#             layers.append(tf.keras.layers.BatchNormalization())
#             layers.append(tf.keras.layers.ReLU())
        
#         layers.append(tf.keras.layers.Conv2DTranspose(filters=nchannel, kernel_size=2, strides=2, padding='same'))
#         layers.append(tf.keras.layers.BatchNormalization())
#         layers.append(tf.keras.layers.Activation(activation='sigmoid'))  
                
#         self.units = tf.keras.Sequential(layers)     

#     def call(self, inputs):
#         decoded = self.units(inputs)
#         return decoded

    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, kernels, name="encoder"):
        super(Encoder, self).__init__(name=name)
        nElements = 9 
        
        layers = []
        for kernel in kernels:
            layers.append(tf.keras.layers.Conv2D(filters=kernel, kernel_size=2, strides=(1, 1), padding='same'))
            layers.append(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
            layers.append(tf.keras.layers.ReLU())
        
        layers.append(tf.keras.layers.Conv2D(filters=nElements, kernel_size=1, strides=(1, 1), padding='same'))
        self.units = tf.keras.Sequential(layers)
            
    def call(self, inputs):
        encoded = self.units(inputs)
        return encoded

class Reconstruction(tf.keras.layers.Layer):
    def __init__(self, kernels, nchannel, name="reconstruction"):
        super(Reconstruction, self).__init__(name=name)
        
        layers = []
        layers.append(tf.keras.layers.Conv2D(filters=kernels[0], kernel_size=1, strides=(1, 1), padding='same'))

        for kernel in kernels[1:]:
            layers.append(tf.keras.layers.Conv2DTranspose(filters=kernel, kernel_size=2, strides=2, padding='same'))
            layers.append(tf.keras.layers.ReLU())
        
        layers.append(tf.keras.layers.Conv2DTranspose(filters=nchannel, kernel_size=2, strides=2, padding='same'))
        # layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation(activation='sigmoid'))  
                
        self.units = tf.keras.Sequential(layers)     

    def call(self, inputs):
        decoded = self.units(inputs)
        return decoded