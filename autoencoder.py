from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
Dense, Flatten, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
import numpy as np

class Autoencoder:
    '''
    Autoencoder represents a Deep Convolutional autoencoder architecture with mirrored
    encoder and decoder components.
    
    '''
    
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        self.input_shape  = input_shape  #ex) [width, height, channel]
        self.conv_filters = conv_filters #ex) [2,4,8]: number of conv_filter for each convolutional layer
        self.conv_kernels = conv_kernels #ex) [3,5,4]: filter size of each conv layer
        self.conv_strides = conv_strides #ex) [1,2,2]: stride size 
        self.latent_space_dim = latent_space_dim #int: number of dim 
        
        self.encoder = None
        self.decoder = None
        self.model   = None
        
        #private attributes
        self._num_conv_layers = len(conv_filters) 
        self._shape_before_bottleneck = None
        
        #construct model 
        self._build()
        
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        
    def _build(self):
        self._build_encoder()
        self._build_decoder()
#         self._build_autoencoder()
        
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name = "decoder")
        
    def _add_decoder_input(self):
        return Input(shape = self.latent_space_dim, name = "decoder_input")
    
    def _add_dense_layer(self, decoder_input):
        #we want to have same number of neurons before the bottleneck process
        num_neurons = np.prod(self._shape_before_bottleneck) # [width, height, channel] -> need their product
        dense_layer = Dense(num_neurons, name = "decoder_dense")(decoder_input)
        
        return dense_layer
        
    def _add_reshape_layer(self, dense_layer):
        #reshaping to 3D block
        return Reshape(target_shape = self._shape_before_bottleneck)(dense_layer)
    
    def _add_conv_transpose_layers(self, x):
        '''
        Add conv transpose blocks
        Loop thru all the encoder's conv layer in reverse order and stop at the first layer
        '''
        
        for layer_index in reversed(range(1,self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
            
        return x
    
    def _add_conv_transpose_layer(self, layer_index, x):
        
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index], 
            strides = self.conv_strides[layer_index],
            padding = "same", 
            name = "decoder_conv_transpose_layer_{}".format(layer_num)
        )
        
        x = conv_transpose_layer(x)
        x = ReLU(name = "decoder_relu_{}".format(layer_num))(x)
        x = BatchNormalization(name = "decoder_bn_{}".format(layer_num))(x)
        
        return x
        
    def _add_decoder_output(self, x):
        
        conv_transpose_layer = Conv2DTranspose(
            filters = 1,
            kernel_size = self.conv_kernels[0], 
            strides = self.conv_strides[0],
            padding = "same", 
            name = "decoder_conv_transpose_layer_{}".format(self._num_conv_layers)
        )    
        
        x = conv_transpose_layer(x)
        output_layer = Activation(activation = "sigmoid", name = "sigmoid_layer")(x)
        
        return output_layer
    
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottlenext(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name = "encoder") #input/output of the encoder 
        
    def _add_encoder_input(self):
        return Input(shape = self.input_shape, name = "encoder_input")
    
    def _add_conv_layers(self,encoder_input):
        '''
        Creates all convolutional layers needed for encoder
        '''
        x = encoder_input
        
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x) #iteratively adding layer to nn graph
            
        return x
        
    def _add_conv_layer(self, layer_index, x):
        '''
        Add convolutional block to a graph of layers consisting of conv2_d + Relu + batch normalization
        '''
        layer_num = layer_index + 1
        
        conv_layer = Conv2D(
            filters = self.conv_filters[layer_index], 
            kernel_size = self.conv_kernels[layer_index], 
            strides = self.conv_strides[layer_index], 
            padding = "same", 
            name = "encoder_conv_layer_{}".format(layer_num)
        )
        
        x = conv_layer(x) #using Functional api
        x = ReLU(name = "encoder_relue_{}".format(layer_num))(x)
        x = BatchNormalization(name = "encoder_bn_{}".format(layer_num))(x)
        
        return x 

    def _add_bottlenext(self, x):
        '''
        Flatten input and attach Dense layer as bottleneck
        return encoder output, shape of the output before flatten for decoder
        '''
    
        # Returns the shape of tensor or variable as a list of int or NULL entries.
        # shape of data (4-dim array): [batch_size, width, height, channel] 
        self._shape_before_bottleneck = K.int_shape(x)[1:] 
        
        x = Flatten()(x)
        x = Dense(units = self.latent_space_dim, name = "encoder_output")(x)
        
        return x 


if __name__ == "__main__":
    #initiating autoencoder
    autoencoder = Autoencoder(input_shape = (28,28,1),
                              conv_filters=(32,64,64,64), 
                              conv_kernels = (3, 3, 3, 3), 
                              conv_strides=(1, 2, 2, 1), 
                              latent_space_dim= 2
                             )
    
    autoencoder.summary()

# need to work on combining encoder and decoder
