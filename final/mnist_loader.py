import numpy as np
import os.path
NUM_SAMPLES = 100 # number of samples to include <= 10000

from mnist import MNIST
mndata = MNIST(os.path.abspath('dataset'))
images,labels = mndata.load_testing()

# Get a numpy array of images
def get_numpy_array(num_samples=NUM_SAMPLES):
    arr = []
    for i in range(0,num_samples):
        image = []
        row = []
        for j in range(0,len(images[i])):
            pixl = images[i][j]
            #print(pixl)
            if pixl == 0:
                row.append(0)
            else:
                row.append(1)
            if j%28 == 27 and len(row) > 0 :
                image.append(row)
                row = []
        arr.append(np.array(image))
        #print(len(image))
    #print('retrieved',len(arr))
    return arr

if __name__ == "__main__":
    print('run test')
    images = get_numpy_array()
    print(images[0],len(images),len(images[0]),len(images[0][0]))