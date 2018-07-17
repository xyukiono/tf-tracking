# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

def tile_images(data, padsize=1, padval=0):
    """
    Convert an array with shape of (B, C, H, W) into a tiled image.
    Copy from DeLorean code
    """
    assert(data.ndim == 4)
    data = data.transpose(0, 2, 3, 1)

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (
        (0, n ** 2 - data.shape[0]),
        (0, padsize),
        (0, padsize)
    ) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(
        data, padding, mode='constant', constant_values=(padval, padval))
    data = data.reshape(
        (n, n)
        + data.shape[1:]
    ).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    if data.shape[2] == 1:
        # Return as (H, W)
        return data.reshape(data.shape[:2])
    return data

def plot_images(images, cls_true, class_names=None, cls_pred=None, smooth=True, num_plots=9):

    assert len(images) == len(cls_true)
    num_plots = min(len(images), num_plots)

    num_grids = int( math.ceil(math.sqrt(num_plots)) )
    
    # Create figure with sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < num_plots:
            # Interpolation type.
            if smooth:
                interpolation = 'spline16'
            else:
                interpolation = 'nearest'

            # Plot image.
            if images.ndim == 4:
                ax.imshow(images[i, :, :, :],
                          interpolation=interpolation)
            else:
                ax.imshow(images[i, :, :],
                          interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]] if (class_names!=None) else '{0}'.format(cls_true[i]) 

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]] if (class_names!=None) else '{0}'.format(cls_pred[i]) 

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_example_errors(images, cls_true, cls_pred, correct, num_plots=9):
     # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_true[incorrect]
    
    # Plot the first 9 images.
    num_plots = min(num_plots, len(images))
    plot_images(images=images[0:num_plots],
                cls_true=cls_true[0:num_plots],
                cls_pred=cls_pred[0:num_plots], num_plots=num_plots)

def print_conv_weights(weights, input_channel=0):
    # Assume weights are TensorFlow ops for 4-dim variables
    # e.g. weights_conv1 or weights_conv2.

    # Retrieve the values of the weight-variables from TensorFlow.
    # A feed-dict is not necessary because nothing is calculated.
    # weights can be obtained by session.run(weights_simbol)

    # Print statistics for the weights.
    print("Min:  {0:.5f}, Max:   {1:.5f}".format(weights.min(), weights.max()))
    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(weights.mean(), weights.std()))
    
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)
    abs_max = max(abs(w_min), abs(w_max))

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # The format of this 4-dim tensor is determined by the
            # TensorFlow API. See Tutorial #02 for more details.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=-abs_min, vmax=abs_max,
                      interpolation='nearest', cmap='seismic')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()    

def plot_layer_output(layer_output):
    # Assume layer_output is a 4-dim tensor
    # e.g. output_conv1 or output_conv2.

    # Create a feed-dict which holds the single input image.
    # Note that TensorFlow needs a list of images,
    # so we just create a list with this one image.

    # layer_output = session.run(layer_output_simbol, feed_dict={x: [image]})

    # Get the lowest and highest values.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    output_min = np.min(layer_output)
    output_max = np.max(layer_output)

    # Number of image channels output by the conv. layer.
    num_images = layer_output.shape[3]

    # Number of grid-cells to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = int(math.ceil(math.sqrt(num_images)))
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid image-channels.
        if i<num_images:
            # Get the images for the i'th output channel.
            img = layer_output[0, :, :, i]

            # Plot image.
            ax.imshow(img, vmin=output_min, vmax=output_max,
                      interpolation='nearest', cmap='binary')
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def print_confusion_matrix(cls_true, cls_pred):

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')




class SummaryPlotter():
    class PlotData():
        def __init__(self, xdata, ydata, label):
            self.xdata = xdata
            self.ydata = ydata
            self.label = label
            
    def __init__(self):
        self.reset()
        
    def add_data(self, title, xdata, ydata, label):
        if title not in self.data:
            self.data[title] = []
        self.data[title].append(SummaryPlotter.PlotData(xdata, ydata, label))
    def reset(self):
        self.data = OrderedDict()
        self.axes_properties = {}
        self.hspace = None
        self.wspace = None
        self.figsize=(12, 6)
        
    def set_space(self, hspace=None, wspace=None):
        self.hspace = hspace
        self.wspace = wspace
        
    def set_axes_property(self, title, prop_name, prop_value):
        if title not in self.axes_properties:
            self.axes_properties[title] = []
        self.axes_properties[title].append( (prop_name, prop_value) )
        
    def plot_all(self, num_cols = 0, num_rows = 0, save_path=None):
        num_data = len(self.data)
        if num_data == 0:
            return
        if num_data > num_cols * num_rows:
            num_cols = 1 # y
            num_rows = num_data # x 
        fig, axes = plt.subplots(num_cols, num_rows, figsize=self.figsize)
        
        fig.subplots_adjust(hspace=self.hspace, wspace=self.wspace)
        
        axes_flat = axes.flat
        num_fig = len(axes_flat)

        for i, (title, pltdata_list) in enumerate(self.data.items()): 
            self.plot(title, ax=axes_flat[i])
        if num_fig > num_data:
            for i in range(num_data, num_fig):
                axes_flat[i].set_xticks([])
                axes_flat[i].set_yticks([])

        if save_path is not None:
            plt.savefig(save_path)
            
    def plot(self, title, ax=None):
        if title not in self.data:
            return
        if ax is None:
            ax = plt.gca()
        pltdata_list = self.data[title]
        for data in pltdata_list:
            ax.plot(data.xdata, data.ydata, label=data.label)
        ax.set_title(title)
        ax.legend()
        if title in self.axes_properties:
            for (name, value) in self.axes_properties[title]:
                getattr(ax, name)(value)    
