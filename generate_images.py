from PIL import ImageEnhance
from PIL import Image
import numpy as np
import random
import glob


#see PIL/pillow documentation for what these mean
output_shape = (200,200)
rotate = (0,360)
contrast = (.7,1.3)
brightness = (.7,1.3)
color = (.7,1.3)

def make_more_images(image,new_filename,category):
    '''
    Given an image apply a random set of transformations with random values to
    make a new image. This allows us to massively increase the amount of images
    to train on. I'm choosing to store the image label in the filename for now,
    but this might change.
    '''
    im = Image.open(image,mode='r')
    
    #determine which transformations to apply at random
    truth = [random.getrandbits(1) for i in range(5)]
    im_new = im


    #rotate image
    if truth[0]:
        rotation_param = np.random.randint(rotate[0],rotate[1])
        im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR)
        im_new = im_new.crop(im_new.getbbox())
        im_new = im_new.resize(output_shape, resample=Image.BICUBIC)

    #tweak contrast
    if truth[1]:
        contrast_param = np.random.uniform(contrast[0], contrast[1])
        im_new = ImageEnhance.Contrast(im_new).enhance(contrast_param)

    #tweak brightness
    if truth[2]:
        brightness_param = np.random.uniform(brightness[0],brightness[1])
        im_new = ImageEnhance.Brightness(im_new).enhance(brightness_param)

    #tweak color
    if truth[3]:
        color_param = np.random.uniform(color[0], color[1])
        im_new = ImageEnhance.Color(im_new).enhance(color_param)

    #flip image
    if truth[4]:
        im_new = im_new.transpose(Image.FLIP_LEFT_RIGHT)
    im_new.save("img_{count}_{category}.jpg".format(count = new_filename,category = category))

def main():
    '''
    Generate a set number of new images. An image is selected at random from the 
    set of images which we were given and then a random alteration is performed and
    the image is saved to a file.
    '''
    #replace with the path to your original bee images
    images = glob.glob('../bee_images/train/*.jpg')
    max_idx = len(images)
    bee_dict = dict()
    images_to_gen = 50000

    #we need to find out classification information to apply to our new images
    #replace with the path to your csv
    with open('../bee_classification.csv','rt') as f:
        for line in f:
            line = line.rstrip()
            row = line.split(',')
            bee_dict[row[0]] = row[1]

    for i in range(images_to_gen):
        file_name = images[np.random.randint(0,max_idx)]
        
        #get index of the image
        idx_start = file_name.find('/train/')+len('/train/')
        idx_finish = file_name.find('.jpg')
        idx = file_name[idx_start:idx_finish]
        
        make_more_images(file_name,i,bee_dict[idx])

if __name__=="__main__":
    main()
