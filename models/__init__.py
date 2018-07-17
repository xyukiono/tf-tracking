from .vgg import vgg_16
from .resnet_v2 import resnet_v2_50
from .mobilenet_v2 import mobilenet_v2
from .alexnet import alexnet
import tensorflow as tf

IMAGENET_RGB_MEAN = [123.68, 116.78, 103.94] # R,G,B order
# resnet_v2_50.R_MEAN = 123.68
# resnet_v2_50.G_MEAN = 116.78
# resnet_v2_50.B_MEAN = 103.94
# resnet_v2_50.RGB_MEAN = [resnet_v2_50.R_MEAN, resnet_v2_50.G_MEAN, resnet_v2_50.B_MEAN]

def preprocess_images(images, height, width, mean):
    images = tf.image.resize_images(images, (height, width))
    images = images - mean
    return images