all:
	/usr/local/cuda/bin/nvcc --std=c++11 -Werror cross-execution-space-call -lm  -o solution lab2.cu -lGL -lGLU -lglut -lGLEW 
clean:
	rm -rf solution
