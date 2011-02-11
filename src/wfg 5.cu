
/*
 * Author: Henry Peng (20165483)
 * University of Western Australia
 * School of Computer Science and Software Engineering
 * December 2010
 */

//#####################################
// Strategy for possible improvement:
//#####################################
// 1) Reduce Memcpy calls needed: Memcpy whole fr[iteration] to global variable ?
// 2) Reduce the number variables necessary for device (extra cudamallocing may reduce speed). e.g. remove temp variable
// 3) Parallel prefix sum, reduce number of threads needed? Can it increase speed? Currently: 256 threads
// 4) Reduce memory required for eliminated array and scan results array. Currently fixed at 512 points for each. plus Not scalable.
// 5) Reduce the size of memory allocated for each cuda variable.

// 6) Make into single kernel function with many device codes, single cudaThreadsynchronise (implement scan_best in device mode)
// 7) Look into using shared memory
//######################################

/////////////////////////////////////////////////////////
// Includes and Defines
/////////////////////////////////////////////////////////

//#define CUDPP_STATIC_LIB
#include "/usr/local/NVIDIA_GPU_Computing_SDK/C/common/inc/cudpp/cudpp.h"
#include "read.c"
#include "scan_best_kernel.cu"
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>
#include "cutil.h"
//#include "radixsort.cu"

//extern float ehv(int index, FRONT front);
//extern float hypervolume(FRONT front);

#define MIN(x, y) ((x < y) ? (x) : (y))
unsigned int frontSize;
unsigned int pointSize;

/////////////////////////////////////////////////////////
// Global Variables
/////////////////////////////////////////////////////////

//int maxDepth = -1;	//the maximum depth you have reached
int n = 0; //the dimension of the current front we are working on
int iteration = 0;	//depth of the recursion starting from 0
float hypervolume(FRONT);
FRONT *fr;	//storage for storing array of sprimed/non-dominating fronts as we go deeper into the recursion

/* new hypervolume variables */
FRONT *fs;      // the stack of fronts
int *ps;        // the indices of the contributing points
float *vs;     // partial volumes
float *evs;    // exclusive volumes

/* Device global variables */
float *d_temp;
float *d_front;
int *d_eliminated;
int *d_scanoutput;
float *d_temp2;

/* new hypervolume variables */
FRONT *frontsArray;      // the stack of fronts
int *indexStack;        // the indices of the contributing points
float *hvStack;     // partial volumes
float *ehvStack;    // exclusive volumes

/* new cuda variables */
FRONT *d_frontsArray;
float *d_hvStack;
float *d_ehvStack;

//NOte: n is needed for slicing and sorting, iteration is needed for saving array of fronts when going deeper into recursion

/////////////////////////////////////////////////////////
// GPU Kernel Functions
/////////////////////////////////////////////////////////

/**
 * Sprimes a front in parallel.
 */
__global__ void sprimeFront(float *frPoints_device, float *frontPoints_device, int index) {
	frPoints_device[blockIdx.x*blockDim.x+threadIdx.x] = MIN(frontPoints_device[index*blockDim.x+threadIdx.x], 
	frontPoints_device[(blockIdx.x+1+index)*blockDim.x+threadIdx.x]);
}

/**
 * Device Function: Determine domination status of point A and B.
 * Similar to CPU implementation.
 *
 * returns 1 if point b dominates a
 * zero if non-dominating
 * returns -1 if point a dominates b
 * returns 2 if point is equal
 */
__device__ int dominated(float *point_a, float *point_b, int nDim) {
	int result = 2;
	for (int i = 0; i < nDim; i++) {
		if (point_a[i] > point_b[i]) {
			if (result != 1) result = -1; else return 0;
		} 
		if (point_b[i] > point_a[i]) {
			if (result != -1) result = 1; else return 0;
		}
	}
	return result;
}

/**
 * Computes eliminated array in parallel.
 * Flag = 0 indicates eliminated, flag = 1 is kept.
 * e.g. result of a front with 5 points: [0, 1, 1, 0, 1]. 
 * (known "trivial" bug: equal points will not be eliminated). 
 */
__global__ void computeEliminatedArray(float *d_fr_iteration, int nDim, int *eliminated) {
    	__shared__ int flag;

    	flag = 1;
	__syncthreads();

	if (dominated(&d_fr_iteration[blockIdx.x*nDim] , &d_fr_iteration[threadIdx.x*nDim], nDim) == 1)
		flag = 0;

	__syncthreads();
    
    	//if (threadIdx.x==0) {
    	    eliminated[blockIdx.x] = flag;
	//}
}

/**
 * Insert the results and reorder into temp array in parallel.
 */ 
__global__ void insertResults(float *d_fr_iteration, float *temp, int *eliminated, int *scanoutput) {
	if (eliminated[blockIdx.x] == 1) {
		//insert the non-dominated points
		temp[(scanoutput[blockIdx.x]-1)*blockDim.x+threadIdx.x] = d_fr_iteration[blockIdx.x*blockDim.x+threadIdx.x];
	} /*else {
		//if eliminated insert at the end of the temp array.
		temp[(gridDim.x-1-(blockIdx.x-scanoutput[blockIdx.x]))*blockDim.x+threadIdx.x] = d_fr_iteration[blockIdx.x*blockDim.x+threadIdx.x];
	}*/
}

/**
 * Create an inclusive scan output from exclusive scan output.
 * Shift array left, and insert the sum of last element of scan and 
 * last element of input array, at the end of the sum.
 */ 
__global__ void scan_inclusive(int *d_scanbest, int *d_eliminated, int nPoints) {
	if (threadIdx.x > 0) 
		d_scanbest[threadIdx.x-1] = d_scanbest[threadIdx.x];
	
	//__syncthreads();
	//if (threadIdx.x == nPoints-1) {
	if (nPoints == 1) {
		d_scanbest[nPoints-1] = d_eliminated[nPoints-1];
	} else {
		d_scanbest[nPoints-1] = d_scanbest[nPoints-2] + d_eliminated[nPoints-1];
	}
	//}
}

/////////////////////////////////////////////////////////
// Helper methods
/////////////////////////////////////////////////////////

/**
 * Prefix-sum sequential on CPU. (Deprecated)
 */
int *sequentialScan(int *eliminated, int n)
{
	int *output = (int *) malloc(sizeof(int) *n);
	output[0] = eliminated[0];
	for (int i = 1; i < n; i++) {
		output[i] = output[i-1] + eliminated[i];
	}
	return output;
}

/**
 * Prints a front.
 */
void printfront(FRONT front) {
	for (int j = 0; j < front.nPoints; j++)
	{
		printf("\t");
		for (int k = 0; k < n; k++)
		{
			printf("%f ",front.points[j].objectives[k]);
		}
		printf("\n");
	}
}

/////////////////////////////////////////////////////////
// CPU Functions
/////////////////////////////////////////////////////////

/**
 * Determine domination status of point A and B.
 *
 * returns 1 if point b dominates a
 * zero if non-dominating
 * returns -1 if point a dominates b
 * returns 2 if point is equal
 */
int dominated(POINT a, POINT b) {
	int result = 2;
	for (int i = 0; i < n; i++) {
		if (a.objectives[i] > b.objectives[i]) {
			if (result != 1) result = -1; else return 0;
		}
		if (b.objectives[i] > a.objectives[i]) {
			if (result != -1) result = 1; else return 0;
		}
	}
	return result;
}

/**
 * compare function for qsort sorting front in the last objective, i.e. increasing from top to bottom
 * and we process hypervolumes from the bottom 
 */
int compare (const void *a, const void *b)
{
	//n == maxDimensions-iteration
	for (int i = n - 1; i >= 0; i--) {
		if (((*(POINT *)a).objectives[i] > (*(POINT *)b).objectives[i])) return 1;
		if (((*(POINT *)a).objectives[i] < (*(POINT *)b).objectives[i])) return -1;
	}
	return 0;
}

/**
 * Returns the size of exclusive hypervolume of point p at index relative to a front set.
 */
float ehv(int index, FRONT front) {
	
	//hypervolume of a single poinit
	float ehv = 1;
	for (int i = 0; i < n; i++) {
		ehv *= front.points[index].objectives[i];
	}
	
	//if not the last point, then go deeper into the recursion
	if (index < front.nPoints-1) {
		limitset(index, front);		//limit the front relative to index.
		iteration++;	//slicing
		ehv -= hypervolume(fr[iteration-1]);	//subtract the hypervolume of the limit set from ehv.
		iteration--;
	}
	
	return ehv;
}

/**
 * Returns the size of hypervolume of a front.
 */
float hypervolume(FRONT front) {
	//sort the front with qsort
	qsort(front.points, front.nPoints, sizeof (POINT), compare);
	
	//calculate for base case = 2D
	if (n==2) {
		float vol2d = (front.points[0].objectives[0] * front.points[0].objectives[1]);
		for (int i = 1; i < front.nPoints; i++) {
			vol2d += (front.points[i].objectives[0]) * 
						   (front.points[i].objectives[1] - front.points[i - 1].objectives[1]);
		}
		return vol2d;
	}
	
	float sumhv = 0;
	n--;
	//sum all the segments
	for (int i = front.nPoints - 1; i >= 0; i--)
		//for (int i = 0; i < front.nPoints; i++) //annoying bug that cause inaccurate results
		sumhv += front.points[i].objectives[n] * ehv(i, front);
	n++;
	
	return sumhv;
}

// creates the front frontsArray[fr-1].points[indexStack[fr-1]+1 ..] in frontsArray[fr], 
// with each point bounded by frontsArray[fr-1].points[indexStack[fr-1]] 
// and with dominated points removed 
void makeDominatedBit()
{

  	int z = frontsArray[iteration-1].nPoints - 1 - indexStack[iteration-1];
	for (int i = 0; i < z; i++) {
		for (int j = 0; j < n; j++) {
			frontsArray[iteration].points[i].objectives[j] = MIN(frontsArray[iteration-1].points[indexStack[iteration-1]].objectives[j],
                                     frontsArray[iteration-1].points[indexStack[iteration-1] + 1 + i].objectives[j]);
		}
	}
  
	POINT t; // have to do proper swaps because of the reuse of the memory hierarchy 
  	frontsArray[iteration].nPoints = 1;
  	for (int i = 1; i < z; i++) {
		int j = 0;
   		bool keep = true;

    		while (j < frontsArray[iteration].nPoints && keep) {
       			switch (dominated(frontsArray[iteration].points[i], frontsArray[iteration].points[j])) {
				case -1: 
					t = frontsArray[iteration].points[j];
					frontsArray[iteration].points[j] = frontsArray[iteration].points[frontsArray[iteration].nPoints - 1]; 
					frontsArray[iteration].points[frontsArray[iteration].nPoints - 1] = t; 
					frontsArray[iteration].nPoints--; 
					break;
          			
				case  0: 
					j++; 
					break;
          
					// case  2: printf("Identical points!\n");
	 			default: 
					keep = false;
			}
		}
     		
		if (keep) {
			t = frontsArray[iteration].points[frontsArray[iteration].nPoints]; 
			frontsArray[iteration].points[frontsArray[iteration].nPoints] = frontsArray[iteration].points[i]; 
			frontsArray[iteration].points[i] = t; 
			frontsArray[iteration].nPoints++;	
		}
    	}
}

void hvnew() {
	// sets hvStack[0] to the hypervolume of frontsArray[0][0 ..] 
  	qsort(frontsArray[0].points, frontsArray[0].nPoints, sizeof(POINT), compare);

  	indexStack[0] = frontsArray[0].nPoints - 1;

  	while (indexStack[0] >= 0) { // there are jobs remaining 
    		if (indexStack[iteration] < 0) {	// we've finished the jobs at this level: i.e. completed all ehv calculation (HV is complete for that level!)
			iteration--; 
			// compute the single point ehv excluding the last objective
      			ehvStack[iteration] -= hvStack[iteration+1]; 
			//  add the ehv multiplied by the last objective left out due to n--, to the hv stack
      			hvStack[iteration] += (frontsArray[iteration].points[indexStack[iteration]].objectives[n]) * ehvStack[iteration];
			// 1 job is finished for the previous iteration
       			indexStack[iteration]--;
			// finished with next level ehv
       			n++;
		} else if (n == 2) {  	// do this job using the linear algorithm 
			//TODO make this work
			/*if (indexStack[0] == 0) { //or iteration== 0
				hvStack[0] = frontsArray[0].points[0].objectives[0] * frontsArray[0].points[0].objectives[1];
				for (int i = 1; i < frontsArray[0].nPoints; i++) {
					hvStack[0] += (frontsArray[0].points[i].objectives[0]) * 
								   (frontsArray[0].points[i].objectives[1] - frontsArray[0].points[i - 1].objectives[1]);
				}
				indexStack[0]--;
				n++;
			} else {*/
			iteration--;
       			ehvStack[iteration] -= frontsArray[iteration+1].points[0].objectives[0] * frontsArray[iteration+1].points[0].objectives[1]; 
       			for (int i = 1; i < frontsArray[iteration+1].nPoints; i++) {
         			ehvStack[iteration] -= (frontsArray[iteration+1].points[i].objectives[0]) * (frontsArray[iteration+1].points[i].objectives[1] - frontsArray[iteration+1].points[i-1].objectives[1]);
			}
       			hvStack[iteration] += frontsArray[iteration].points[indexStack[iteration]].objectives[n] * ehvStack[iteration];
      			indexStack[iteration]--;
       			n++;
			//}
		} else {  // we need to "recurse" 
      			n--;

       			ehvStack[iteration] = 1;
       			for (int i = 0; i < n; i++)  {
				//compute the single point ehv excluding the last objective
         			ehvStack[iteration] *= frontsArray[iteration].points[indexStack[iteration]].objectives[i];
			}

       			if (indexStack[iteration] == frontsArray[iteration].nPoints - 1) { 	// first job at this level: set will be empty = no need to recurse
				// add the first ehv multiplied by the last objective left out due to n--, to the hv stack
        			hvStack[iteration] = frontsArray[iteration].points[indexStack[iteration]].objectives[n] * ehvStack[iteration];
          			indexStack[iteration]--;
				// finished with first level ehv (index = nPoints-1), now need to calculate the levels until reach (index = 0)
          			n++;
			} else { // set will be non-empty: create a new job 
				//go to next level recursion
         			iteration++; 
          			makeDominatedBit(); 
          			qsort(frontsArray[iteration].points, frontsArray[iteration].nPoints, sizeof(POINT), compare);
				//reset index stack to the number of points-1
          			indexStack[iteration] = frontsArray[iteration].nPoints - 1;
			}
		}
	}
}

/**
 *  Timer Functions
 */
void run(int argc, char *argv[])
{
    unsigned int timer = 0;

    CUT_DEVICE_INIT(argc, argv);

    /////////////////////////////////////////////////////////////////////
    // Create and start a timer called "timer"
    // alls to create ans start times are enveloped in the CUT_SAFE_CALL
    // This CUDA Utility Tool checks for errors upon return.
    // If an error is found, it prints out and error message, file name,
    // and line number in file where the error can be found
    /////////////////////////////////////////////////////////////////////
    timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));
    
    // Stop the timer
    CUT_SAFE_CALL(cutStopTimer(timer));
    printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer));

    // Delete the timer
    CUT_SAFE_CALL(cutDeleteTimer(timer));
}

/**
 * Runs a parallel hypervolume
 */ 
__global__ void hvparallellol() {
	// Should call many device functions

	//sortParallel();

	//d_indexStack[0] = d_frontsArray[0].nPoints - 1;
	
}

/**
 * Runs a parallel hypervolume
 */
//void hvparallel() {
	/*int blockSize = 100;
	int nBlocks = N/blockSize + (N%blockSize == 0 ? 0:1);
	// where N is the parallel threads required

	global<<<nBlocks, blockSize>>> ( param , N );

	cudaThreadSynchronize();
	checkCUDAError("HV parallel failed!");*/
//}

////////////////////////////////////////////////////////////////
// CUDA Helpers
////////////////////////////////////////////////////////////////

/**
 * Returns a sprimed & non-dominating front relative to point p at index.
 */
void limitset() {
	// TODO make this a kernel which calls many device functions 

	// sets the number of points in sprimed front
	int z = frontsArray[iteration-1].nPoints - 1 - indexStack[iteration-1];
	
	// sprimes the front and store it into temporary storage
	sprimeFront<<< z, n >>>( d_temp, d_frontsArray, indexStack[iteration-1]);
    	cudaThreadSynchronize();

	// compute eliminated array and store it in d_eliminated
	computeEliminatedArray<<< z, z >>>(d_temp, n, d_eliminated);
	cudaThreadSynchronize();
	
	// compute parallel prefix sum and store the result in d_scanoutput
	// TODO may need to make use of cudpp for this
	scan_best<<< 256, 512/2, sizeof(int)*(512) >>>(d_scanoutput, d_eliminated, 512);
	cudaThreadSynchronize();
	scan_inclusive<<< 1, z >>>(d_scanoutput, d_eliminated, z);  //make the result into an inclusive scan result.
	cudaThreadSynchronize();

	// compute the results and store it in frontArray
	insertResults<<<z,n>>> (d_temp, &frontsArray[iteration], d_eliminated, d_scanoutput);
	cudaThreadSynchronize();

	// update number of points to the host
	cudaMemcpy(&indexStack[iteration], &d_scanoutput[z-1], sizeof(int), cudaMemcpyDeviceToHost); //update number of points
}

/**
 * @param front front to sort
 * @param numElements number of points
 * @param size size of each point
 */
void sortParallel(float *d_points, int numElements, int size) {
	// set the config
	CUDPPConfiguration config;
	config.op = CUDPP_ADD;
	config.datatype = CUDPP_FLOAT;
	config.algorithm = CUDPP_SORT_RADIX;
	config.options = CUDPP_OPTION_FORWARD;
	
	// create the plan
	CUDPPHandle sortPlan = 0;
	CUDPPResult result = cudppPlan(&sortPlan, config, numElements, 1, 0);  

	// if not successful then exit
	if (CUDPP_SUCCESS != result)
	{
		printf("Error creating CUDPPPlan\n");
		exit(-1);
	}

	// Run the sort TODO make the sorting works
	cudppSort(sortPlan, d_out, d_in, numElements);

	// Destroy the plan
	result = cudppDestroyPlan(scanplan);
	if (CUDPP_SUCCESS != result)
	{
		printf("Error destroying CUDPPPlan\n");
		exit(-1);
	}

	// TODO reuse config and destroy plan at the end
}

//////////////////////////////////////////////////////////
//  HV CUDA
//////////////////////////////////////////////////////////

void hvparallel(int nPoints) {
	// sort in parallel
	sortParallel(frontsArray[0].points, nPoints, sizeof(POINT));

	indexStack[0] = nPoints - 1;

	// TODO host cannot access device memory ehv, and hv and frontsArray, need CUDA kernels for this
	while (indexStack[0] >= 0) {
		if (indexStack[iteration] < 0) {
			iteration--; 
      			ehvStack[iteration] = ehvStack[iteration] - hvStack[iteration+1]; 
      			hvStack[iteration] = hvStack[iteration] + (frontsArray[iteration].points[indexStack[iteration]].objectives[n]) * ehvStack[iteration];
       			indexStack[iteration]--;
       			n++;
		} else if (n == 2) {
			iteration--;
       			ehvStack[iteration] -= frontsArray[iteration+1].points[0].objectives[0] * frontsArray[iteration+1].points[0].objectives[1]; 
       			for (int i = 1; i < frontsArray[iteration+1].nPoints; i++) {
         			ehvStack[iteration] -= (frontsArray[iteration+1].points[i].objectives[0]) * 
							(frontsArray[iteration+1].points[i].objectives[1] - frontsArray[iteration+1].points[i-1].objectives[1]);
			}
       			hvStack[iteration] += frontsArray[iteration].points[indexStack[iteration]].objectives[n] * ehvStack[iteration];
      			indexStack[iteration]--;
       			n++;
		} else {
      			n--;
       			ehvStack[iteration] = 1;
       			for (int i = 0; i < n; i++)  {
         			ehvStack[iteration] *= frontsArray[iteration].points[indexStack[iteration]].objectives[i];
			}

       			if (indexStack[iteration] == frontsArray[iteration].nPoints - 1) {
        			hvStack[iteration] = frontsArray[iteration].points[indexStack[iteration]].objectives[n] * ehvStack[iteration];
          			indexStack[iteration]--;
          			n++;
			} else {
         			iteration++; 
          			makeDominatedBit(); 
          			sortParallel(frontsArray[iteration].points, indexStack[iteration], sizeof(POINT));
          			indexStack[iteration]--;
			}
		}
	}
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
	//CUT_DEVICE_INIT(argc, argv);
	
	// read the file
	FILECONTENTS *f = readFile(argv[1]);
	
	// start the timer
	struct timeval tv1, tv2;
	struct rusage ru_before, ru_after;
	getrusage (RUSAGE_SELF, &ru_before);
	
	int maxDimensions = 0;	//the max number of dimensions in the fronts
	int maxPoints = 0;  //the max number of points in the fronts
	
	// find the max number of Points, and the max number of Dimensions
	for (int i = 0; i < f->nFronts; i++) {
		if (f->fronts[i].nPoints > maxPoints) 
			maxPoints = f->fronts[i].nPoints;
		if (f->fronts[i].n > maxDimensions) 
			maxDimensions = f->fronts[i].n;
  	}

	/* allocate for cuda memory */
	cudaMalloc( (void **) &d_temp, maxPoints*maxDimensions*sizeof(float));
	cudaMalloc((void **) &d_scanoutput, (512)*sizeof(int));
	cudaMalloc((void **) &d_eliminated, (512)*sizeof(int));
	cudaMalloc( (void **) &d_front, maxPoints*maxDimensions*sizeof(float));
	cudaMalloc((void**) &d_temp2, sizeof(float)*maxPoints*maxDimensions);
	
	// allocate cuda memory
	frontSize = maxPoints*maxDimensions*sizeof(float);
	pointSize = maxDimensions*sizeof(float);
	cudaMalloc((void **) &d_frontsArray, frontSize * maxDimensions);
	cudaMalloc((void **) &d_ehvStack, sizeof(float) * maxDimensions);
	cudaMalloc((void **) &d_hvStack, sizeof(float) * maxDimensions);

	// allocate cpu memory
	indexStack = (int *) malloc(sizeof(int) * maxDimensions);

	// process each front to get the hypervolumes
	for (int i = 0; i < f->nFronts; i++) {
		// read each front
		FRONT front = f->fronts[i];
		n = front.n;

		// copy front to device memory
		float h_front[front.nPoints*n]; 
		for (int j = 0; j < front.nPoints; j++) {
			for (int k = 0; k < n; k++) {
				h_front[j*n+k] = front.points[j].objectives[k];
			}
		}
		cudaMemcpy(d_frontsArray, h_front, frontSize, cudaMemcpyHostToDevice);		

		// run hv parallel
		hvparallel(front.nPoints);

 		// copy back hvresult
		float hvResult[1];
		cudaMemcpy(hvResult, &d_hvStack[0], sizeof(float), cudaMemcpyDeviceToHost);

		// print them out
		printf("Calculating Hypervolume for Front:%d...\n", i+1);
		printf("\t\t\t\t\t%1.10f\n", hvResult);
	}
	
	// stop timer
	getrusage (RUSAGE_SELF, &ru_after);
	tv1 = ru_before.ru_utime;
	tv2 = ru_after.ru_utime;
	printf("Average time = %fs\n", (tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6) / f->nFronts);

	// TODO free the storage

	return 0;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(-1);
    }                         
}

