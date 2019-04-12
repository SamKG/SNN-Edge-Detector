NUM_SAMPLES = 100 # number of samples to include <= 10000

from mnist import MNIST
mndata = MNIST('dataset')
images,labels = mndata.load_testing()

def get_numpy_array():
    arr = []
    for i in range(0,NUM_SAMPLES):
        arr.append(images[i])
    return arr
