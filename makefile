SRC = src
CFILE = wfg\ 12_iterative.c
CUFILE = wfg\ 5.cu
BIN = bin/read 

#############
#############

FILES = $(SRC)/$(CUFILE) $(SRC)/$(CFILE) $(SRC)/scan_best_kernel.cu
compile: $(SRC)/read.c $(SRC)/wfg.h $(FILES)
	gcc -Wall -Werror -pedantic -std=c99 -o $(BIN) $(SRC)/$(CFILE)
	scp $(FILES) 20165483@uggp.csse.uwa.edu.au:~/Desktop/GPU/WFG
