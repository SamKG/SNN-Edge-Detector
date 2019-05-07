class Pixel:
	def __init__(self, pos, threshold = 0.5):
		self.pos = pos
		self.threshold = threshold
		self.color = 

class PixelGrid:
	def __init__(self, neurongrid, threshold):
		self.pixels = np.array([[0 for c in range(len(neurongrid[0]))] for r in range(len(neurongrid))])
		for row in neurongrid:
			for neuron in row:
				pixlscale = 