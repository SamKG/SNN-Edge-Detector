class Clock:
	def __init__(self, dt):
		self.prevtime = 0
		self.currtime = 0
		self.dt = dt
	def start(self):
		self.prevtime = 0
		self.currtime = 0
	def tick(self):
		self.prevtime = self.currtime
		self.currtime += self.dt
	def deltatime(self):
		return self.currtime-self.prevtime
	def get_time(self):
		return self.currtime
	def get_timespan(self):
		return [i*dt for i in range(0,int(self.currtime/self.dt))]