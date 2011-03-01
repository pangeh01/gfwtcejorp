SRC = src
CFILE = wfg\ 12_iterative.c
CUFILE = wfg\ 7.cu
BIN = bin/read 
QS  = src/qsort.cu
TEST = src/test1.cu

#############
#############

FILES = $(SRC)/wfg.h $(SRC)/$(CUFILE) $(SRC)/$(CFILE) $(SRC)/scan_best_kernel.cu $(SRC)/radixsort.cu
compile: $(SRC)/read.c $(SRC)/wfg.h $(FILES)
	gcc -Wall -Werror -pedantic -std=c99 -o $(BIN) $(SRC)/$(CFILE)
	scp $(FILES) 20165483@uggp.csse.uwa.edu.au:~/Desktop/GPU/WFG
	scp $(QS) 20165483@uggp.csse.uwa.edu.au:~/Desktop/GPU/WFG
	 scp "src/wfg (working but slow).cu" 20165483@uggp.csse.uwa.edu.au:~/Desktop/GPU/WFG

emu:
	scp $(TEST) 20165483@uggp.csse.uwa.edu.au:~/Desktop/GPU/WFG
