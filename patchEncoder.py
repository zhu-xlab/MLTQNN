import tensorflow as tf
import numpy as np


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, img_shape, name="patch"):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_shape[0]
        self.img_channel = img_shape[-1]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, 8, 8, self.patch_size, self.patch_size, self.img_channel])
        return patches
    
    

class Encoder(tf.keras.layers.Layer):
    def __init__(self, kernels, name="encoder"):
        super(Encoder, self).__init__(name=name)
        nElements = 9 
        
        layers = []
        for kernel in kernels:
            layers.append(tf.keras.layers.Conv2D(filters=kernel, kernel_size=2, strides=(1, 1), padding='same'))
            layers.append(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
            layers.append(tf.keras.layers.ReLU())
        
        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.Dense(nElements))

        self.units = tf.keras.Sequential(layers)
            
    def call(self, inputs):
        outputs = []
        for i in range(inputs.shape[1]):
            temp_outputs = []
            for j in range(inputs.shape[2]):
                encoded = self.units(inputs[:, i, j, :, :, :])
                temp_outputs.append(encoded)
            outputs.append(tf.stack(temp_outputs, 1))
        return tf.stack(outputs, 1)
    

class Decoder(tf.keras.layers.Layer):
    def __init__(self, kernels, nchannel, name="decoder"):
        super(Decoder, self).__init__(name=name)
        
        self.dense = tf.keras.layers.Dense(2*2*kernels[0], activation='relu')
        self.kernels = kernels[0]
        
        layers = []
        if len(kernels)>1:        
            for kernel in kernels[1:]:
                layers.append(tf.keras.layers.Conv2DTranspose(filters=kernel, kernel_size=2, strides=2, padding='same'))
                layers.append(tf.keras.layers.ReLU())
        
            layers.append(tf.keras.layers.Conv2DTranspose(filters=nchannel, kernel_size=2, strides=2, padding='same'))
            layers.append(tf.keras.layers.Activation(activation='sigmoid'))  
        else:
            layers.append(tf.keras.layers.Conv2DTranspose(filters=nchannel, kernel_size=2, strides=2, padding='same'))
            layers.append(tf.keras.layers.Activation(activation='sigmoid'))              
        self.units = tf.keras.Sequential(layers)     

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        outputs = []
        for i in range(inputs.shape[1]):
            temp_outputs = []
            for j in range(inputs.shape[2]):
                decoded = self.dense(inputs[:, i, j, :])
                decoded = tf.reshape(decoded, [batch_size, 2, 2, self.kernels])
                decoded = self.units(decoded)
                temp_outputs.append(decoded)
            outputs.append(tf.stack(temp_outputs, 1))          
        return tf.stack(outputs, 1)
    
    
class Unpatch(tf.keras.layers.Layer):
    def __init__(self, patch_shape, image_shape, name="reconstruction"):
        super(Unpatch, self).__init__(name=name)
        self.patch_shape = patch_shape
        self.image_shape = image_shape

    def call(self, patches):
        batch_size = tf.shape(patches)[0]

        image_shape = self.image_shape
        patch_shape = self.patch_shape
        n_channels = image_shape[-1]
        dtype = patches.dtype
        stride = (np.array(patch_shape)).astype(int)
        channel_idx = tf.reshape(tf.range(n_channels), (1, 1, 1, 1, 1, n_channels, 1))
        channel_idx = (tf.ones((batch_size, 8, 8, *patch_shape, 1), dtype=tf.int32) * channel_idx)
        batch_idx = tf.reshape(tf.range(batch_size), (batch_size, 1, 1, 1, 1, 1, 1))
        batch_idx = (tf.ones((batch_size, 8, 8, *patch_shape, 1), dtype=tf.int32) * batch_idx)
        indices = []
        for j in range(8):
            for i in range(8):
                _indices = tf.meshgrid(tf.range(stride[0] * j, patch_shape[0] + stride[0] * j),
                                       tf.range(stride[1] * i, patch_shape[1] + stride[1] * i), indexing='ij')
                _indices = tf.stack(_indices, axis=-1)
                indices.append(_indices)
        indices = tf.reshape(tf.stack(indices, axis=0), (8, 8, *patch_shape[:2], 2))
        indices = tf.repeat(indices[tf.newaxis, ...], batch_size, axis=0)
        indices = tf.repeat(indices[..., tf.newaxis, :], n_channels, axis=-2)
        indices = tf.cast(indices, tf.int32)
        indices = tf.concat([batch_idx, indices, channel_idx], axis=-1)
        images = tf.zeros([batch_size, *image_shape], dtype=dtype)
        outputs = tf.tensor_scatter_nd_update(images, indices, patches)
        return outputs
    

class MultiScaleEncoder(tf.keras.layers.Layer):
    def __init__(self, filters, dialation, name="encoder"):
        super(MultiScaleEncoder, self).__init__(name=name)    
        nElements = 9 
        
        layers = []
        for kernel in filters:
            layers.append(tf.keras.layers.Conv2D(filters=kernel, 
                                                 kernel_size=2, 
                                                 strides=(1, 1), 
                                                 dilation_rate=(dialation, dialation), 
                                                 padding='same'))
            layers.append(tf.keras.layers.MaxPooling2D((2, 2), padding="same"))
            layers.append(tf.keras.layers.ReLU())
        
        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.Dense(nElements))

        self.units = tf.keras.Sequential(layers)
            
    def call(self, inputs):
        outputs = []
        for i in range(inputs.shape[1]):
            temp_outputs = []
            for j in range(inputs.shape[2]):
                encoded = self.units(inputs[:, i, j, :, :, :])
                temp_outputs.append(encoded)
            outputs.append(tf.stack(temp_outputs, 1))
        return tf.stack(outputs, 1)
    
    
class MultiScaleDecoder(tf.keras.layers.Layer):
    def __init__(self, kernels, nchannel, name="decoder"):
        super(MultiScaleDecoder, self).__init__(name=name)
        
        self.dense = tf.keras.layers.Dense(2*2*kernels[0], activation='relu')
        self.kernels = kernels[0]
        
        layers = []
        if len(kernels)>1:        
            for kernel in kernels[1:]:
                layers.append(tf.keras.layers.Conv2DTranspose(filters=kernel, kernel_size=2, strides=2, padding='same'))
                layers.append(tf.keras.layers.ReLU())
        
            layers.append(tf.keras.layers.Conv2DTranspose(filters=nchannel, kernel_size=2, strides=2, padding='same'))
            layers.append(tf.keras.layers.Activation(activation='sigmoid'))  
        else:
            layers.append(tf.keras.layers.Conv2DTranspose(filters=nchannel, kernel_size=2, strides=2, padding='same'))
            layers.append(tf.keras.layers.Activation(activation='sigmoid'))              
        self.units = tf.keras.Sequential(layers)     

    def call(self, inputs):
        input1, input2, input3 = inputs        
        

        batch_size = tf.shape(input1)[0]
        outputs = []
        for i in range(input1.shape[1]):
            temp_outputs = []
            for j in range(input1.shape[2]):
                decoded1 = self.dense(input1[:, i, j, :])
                decoded1 = tf.reshape(decoded1, [batch_size, 2, 2, self.kernels])

                decoded2 = self.dense(input2[:, i, j, :])
                decoded2 = tf.reshape(decoded2, [batch_size, 2, 2, self.kernels])
                
                decoded3 = self.dense(input3[:, i, j, :])
                decoded3 = tf.reshape(decoded3, [batch_size, 2, 2, self.kernels])
                
                decoded = tf.keras.layers.Concatenate()([decoded1, decoded2, decoded3])          
                decoded = self.units(decoded)
                temp_outputs.append(decoded)
            outputs.append(tf.stack(temp_outputs, 1))          
        return tf.stack(outputs, 1)
    

class ImgDecoder(tf.keras.layers.Layer):
    def __init__(self, kernels, nchannel, name="decoder"):
        super(ImgDecoder, self).__init__(name=name)
        
        layers = []
     
        for kernel in kernels:
            layers.append(tf.keras.layers.Conv2DTranspose(filters=kernel, kernel_size=2, strides=2, padding='same'))
            layers.append(tf.keras.layers.ReLU())
        
        layers.append(tf.keras.layers.Conv2DTranspose(filters=nchannel, kernel_size=2, strides=2, padding='same'))
        layers.append(tf.keras.layers.Activation(activation='sigmoid'))  
          
        self.units = tf.keras.Sequential(layers)     

    def call(self, inputs):   
        return self.units(inputs)