# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 19:32:58 2015

@author: sakurai
"""

import os
from StringIO import StringIO
import numpy as np

def load_SAD(which_set='train'):
    DATASET_ROOT_PATH = os.environ['DATA_PATH']
    dataset_name = 'SAD'
    data_file = dict(
        train = 'Train_Arabic_Digit.txt',
        test = 'Test_Arabic_Digit.txt'
    )
    
    dataset_path = os.path.join(DATASET_ROOT_PATH, dataset_name, data_file[which_set])
    
    with open(dataset_path) as f:
        data = []
        for block in f.read().strip().split('\n            \n'):
            sio = StringIO(block)
            array = np.loadtxt(sio)
            data.append(array)
            
        target_list = []
        for i in range(10):
            target_list = target_list + [i] * (len(data) / 10)
        target = np.array(target_list)
        return dict(
            data = data,
            target = target
        )
        
if __name__ == '__main__':
    train = load_SAD('train')
    test = load_SAD('test')