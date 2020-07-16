import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))



def show_random_images(dataset):

	# get some random training images
	dataiter = iter(dataset)
	images, labels = dataiter.next()

	img_list = range(5, 10)

	# show images
	print('shape:', images.shape)
	imshow(torchvision.utils.make_grid(images[img_list]))

def image_show(img, title=None, download_image=None):
        fig = plt.figure(figsize=(7, 7))
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='none')
        if title is not None:
            plt.title(title)
        plt.pause(0.001)



