# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:19:31 2016

@author: gpu2
"""

import caffe
import numpy as np
import h5py

model_file = 'ResNet-50-deploy.prototxt'
pretrained_model = 'ResNet-50-model.caffemodel'

net = caffe.Net(model_file,pretrained_model,caffe.TEST)

f = h5py.File('resnet50.h5','w')


for name in net.params.keys():
    if len(net.params[name])==1:
        print "%s has only one params"%name
        grp = f.create_group(name)
        grp.create_dataset('weights',data=net.params[name][0].data)
        grp.create_dataset('bias',data=np.zeros(shape=(net.params[name][0].data.shape[0],)))
    else:
        grp = f.create_group(name)
        grp.create_dataset('weights',data=net.params[name][0].data)
        grp.create_dataset('bias',data=net.params[name][1].data)

f.close()




