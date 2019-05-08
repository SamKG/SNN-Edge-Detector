all:
	cython --embed -o mnistdetector.c mnistdetector.py 
	gcc -Os -I /usr/include/python3.7m  -o mnistdetector mnistdetector.c  -lpython3.7m -lpthread -lm -lutil -ldl -fopenmp

clean:
	rm mnistdetector mnistdetector.c
