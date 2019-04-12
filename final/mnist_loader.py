import numpy as np
NUM_SAMPLES = 100 # number of samples to include <= 10000

from mnist import MNIST
mndata = MNIST('dataset')
images,labels = mndata.load_testing()

# Get a numpy array of images
def get_numpy_array(num_samples=NUM_SAMPLES):
    arr = []
    for i in range(0,num_samples):
        image = []
        for pixl in images[i]:
            if pixl == 0:
                image.append(0)
            else:
                image.append(1)
        arr.append(np.array(image))
        #print(len(image))
    #print('retrieved',len(arr))
    return arr
