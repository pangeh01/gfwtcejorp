FILE = wfg\ 12_iterative.c
CUFILE = wfg\ 5.cu read.c $(FILE) scan_best_kernel.cu
compile: read.c wfg.h $(FILE) $(CUFILE)
	gcc -Wall -Werror -pedantic -std=c99 -o read $(FILE)
	scp $(CUFILE) 20165483@uggp.csse.uwa.edu.au:~/Desktop/GPU/WFG
