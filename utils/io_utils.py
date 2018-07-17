# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import imghdr

try:
    # sys.version_info.major == 2
    import cPickle as pickle
except:
    # sys.version_info.major == 3
    import pickle

def save_pickle(filename, data, mode='wb'):
    with open(filename, mode) as f:
        pickle.dump(data, f)

def load_pickle(filename, mode='rb', is_py2pickle=False):
    with open(filename, mode) as f:
        if is_py2pickle:
            data = pickle.load(f, encoding='latin1')
        else:
            data = pickle.load(f)
    return data

def find_all_files(path, only_img=False):
    file_list = []
    for (root, dirs, files) in os.walk(path): # searcg recursive
        for file in files: # ファイル名だけ取得
            target = os.path.join(root,file)
            if os.path.isfile(target): # check whether it is a file
                if only_img:
                    if imghdr.what(target) != None:
                        file_list.append(target)
                else:
                    file_list.append(target)
    return file_list

def read_text(filename, dtype=np.float32):
    values = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            if dtype == np.float32:
                values.append(list(map(float, line.split(','))))
            else:
                values.append(line.split(','))
    
    values = np.array(values)
    values = np.squeeze(values)

    return values

def load_images_as_NCHW(img_full_names, read_flag=cv2.IMREAD_COLOR, dtype=np.uint8):
    imgs = []
    for full_name in img_full_names:
        img = cv2.imread(full_name, read_flag)
        if img is None:
            continue
        if img.ndim == 2:
            img = img[None,...]
        else:
            assert(img.ndim == 3)
            img = img.transpose(2,0,1)
            imgs.append(img)

    N = len(imgs)
    if N == 0:
        return None

    # check whether all images have same size
    chw = imgs[0].shape
    for img in imgs:
        assert(img.shape == chw) 
    imgs = np.asarray(imgs, dtype=dtype)
    return img
