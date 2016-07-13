# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 00:10:26 2016

@author: moyan
"""

from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input


def identity_block(x,nb_filter,stage,block,kernel_size=3):
    k1,k2,k3 = nb_filter
    out = Convolution2D(k1,1,1,name='res'+str(stage)+block+'_branch2a')(x)
    out = BatchNormalization(axis=1,name='bn'+str(stage)+block+'_branch2a')(out)
    out = Activation('relu')(out)
    
    out = out = Convolution2D(k2,kernel_size,kernel_size,border_mode='same',name='res'+str(stage)+block+'_branch2b')(out)
    out = BatchNormalization(axis=1,name='bn'+str(stage)+block+'_branch2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3,1,1,name='res'+str(stage)+block+'_branch2c')(out)
    out = BatchNormalization(axis=1,name='bn'+str(stage)+block+'_branch2c')(out)
    
    
    out = merge([out,x],mode='sum')
    out = Activation('relu')(out)
    return out

def conv_block(x,nb_filter,stage,block,kernel_size=3):
    k1,k2,k3 = nb_filter
    
    out = Convolution2D(k1,1,1,name='res'+str(stage)+block+'_branch2a')(x)
    out = BatchNormalization(axis=1,name='bn'+str(stage)+block+'_branch2a')(out)
    out = Activation('relu')(out)
    
    out = out = Convolution2D(k2,kernel_size,kernel_size,border_mode='same',name='res'+str(stage)+block+'_branch2b')(out)
    out = BatchNormalization(axis=1,name='bn'+str(stage)+block+'_branch2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3,1,1,name='res'+str(stage)+block+'_branch2c')(out)
    out = BatchNormalization(axis=1,name='bn'+str(stage)+block+'_branch2c')(out)
    
    x = Convolution2D(k3,1,1,name='res'+str(stage)+block+'_branch1')(x)
    x = BatchNormalization(axis=1,name='bn'+str(stage)+block+'_branch1')(x)
    
    out = merge([out,x],mode='sum')
    out = Activation('relu')(out)
    return out
    
inp = Input(shape=(3,224,224))
out = ZeroPadding2D((3,3))(inp)
out = Convolution2D(64,7,7,subsample=(2,2),name='conv1')(out)
out = BatchNormalization(axis=1,name='bn_conv1')(out)
out = Activation('relu')(out)
out = MaxPooling2D((3,3),strides=(2,2))(out)

out = conv_block(out,[64,64,256],2,'a')
out = identity_block(out,[64,64,256],2,'b')
out = identity_block(out,[64,64,256],2,'c')

out = conv_block(out,[128,128,512],3,'a')
out = identity_block(out,[128,128,512],3,'b')
out = identity_block(out,[128,128,512],3,'c')
out = identity_block(out,[128,128,512],3,'d')

out = conv_block(out,[256,256,1024],4,'a')
out = identity_block(out,[256,256,1024],4,'b')
out = identity_block(out,[256,256,1024],4,'c')
out = identity_block(out,[256,256,1024],4,'d')
out = identity_block(out,[256,256,1024],4,'e')
out = identity_block(out,[256,256,1024],4,'f')

out = conv_block(out,[512,512,2048],5,'a')
out = identity_block(out,[512,512,2048],5,'b')
out = identity_block(out,[512,512,2048],5,'c')

out = AveragePooling2D((7,7))(out)
out = Flatten()(out)
out = Dense(1000,activation='softmax',name='fc1000')(out)

model = Model(inp,out)


model_str = model.to_json()
open('resnet50.json','w').write(model_str)

import h5py

f = h5py.File('resnet50.h5','r')
for layer in model.layers:
    try:
        if layer.name[:3]=='res':
            layer.set_weights([f[layer.name]['weights'][:],f[layer.name]['bias'][:]])
        elif layer.name[:2]=='bn':
            scale_name = 'scale'+layer.name[2:]
            weights = []
            weights.append(f[scale_name]['weights'][:])
            weights.append(f[scale_name]['bias'][:])
            weights.append(f[layer.name]['weights'][:])
            weights.append(f[layer.name]['bias'][:])
            layer.set_weights(weights)
    except Exception:
        print layer.name




