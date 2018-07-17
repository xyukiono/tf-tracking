# -*- coding: utf-8 -*-
import numpy as np
from ipywidgets import interact
from PIL import Image

# [Preparation]
# - check installation of jupyter and ipywidgets with
#   `pip list | grep ipywidgets`
# - make the following jupyter extension enable
#   jupyter nbextension enable --py widgetsnbextension --sys-prefix

def display_image_batch(batch,order_bgr=False, order_nchw=False, global_norm=False):
    # batch.shape = (N,C,H,W)
    N = len(batch)
    min_values = np.zeros(N, dtype=np.float32)
    max_values = np.ones(N, dtype=np.float32) * 255
    normalize = False
    if isinstance(batch, np.ndarray) and np.issubdtype(batch.dtype, np.float):
        if global_norm:
            min_values[:] = batch.min()
            max_values[:] = batch.max()
        else:
            min_values[:] = np.min(batch.reshape(N,-1), axis=1)
            max_values[:] = np.max(batch.reshape(N,-1), axis=1)
        normalize = True
    
    def display_image(idx):
        img = batch[idx].copy()
        if normalize:
            min_value = min_values[idx]
            max_value = max_values[idx]
            if max_value > min_value:
                img = np.clip(255.0/(max_value-min_value) * (img-min_value),0,255).astype(np.uint8)
            else:
                img = np.clip(255.0*(img-min_value),0,255).astype(np.uint8)
        if img.ndim == 3:
            if order_nchw:
                # img.shape = [C,H,W]
                img = img.transpose(1,2,0)
            if img.shape[2] == 3 and order_bgr:
                img[...,[0,1,2]] = img[...,[2,1,0]]
            if img.shape[2] == 1:
                img = img[...,0] # convert [H,W,1] to [H,W]

        return Image.fromarray(img)

    interact(display_image, batch=batch, idx=(0, N-1,1));

#def switch_pylab_notebook():
#    %pylab notebook
#    %pylab notebook # I don't know why but execution twice is fine for system

#def switch_pylab_inline():
#    %pylab inline
