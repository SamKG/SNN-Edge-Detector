NUM_SAMPLES = 100 # number of samples to include <= 10000

from mnist import MNIST
mndata = MNIST('dataset')
images,labels = mndata.load_testing()

# Get a numpy array of images
def get_numpy_array(num_samples=NUM_SAMPLES):
    arr = []
    for i in range(0,num_samples):
        arr.append(images[i])
    return arr
