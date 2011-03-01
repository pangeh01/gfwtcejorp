
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

/* Device global variables */
float *d_temp;
float *d_front;
int *d_eliminated;
int *d_scanoutput;
float *d_temp2;

// unused anymore
FRONT *frontsArray;
float *ehvStack;
float *hvStack;

// cpu memory stacks
int *indexStack;
int *nPointsStack;

// cuda memory stacks
float *d_frontsArray;
float *d_hvStack;
float *d_ehvStack;

//NOte: n is needed for slicing and sorting, iteration is needed for saving array of fronts when going deeper into recursion

/////////////////////////////////////////////////////////
// GPU Kernel Functions
/////////////////////////////////////////////////////////

/**
 * Sprimes a front in parallel.
 */
__global__ void sprimeFront(float *frPoints_device, float *frontPoints_device, int index, int pointSize) {
	frPoints_device[blockIdx.x*pointSize+threadIdx.x] = MIN(frontPoints_device[index*pointSize+threadIdx.x], 
	frontPoints_device[(blockIdx.x+1+index)*pointSize+threadIdx.x]);
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
__global__ void computeEliminatedArray(float *d_fr_iteration, int nDim, int *eliminated, int pointSize) {
    	__shared__ int flag;

    	flag = 1;
	__syncthreads();

	if (dominated(&d_fr_iteration[blockIdx.x*pointSize] , &d_fr_iteration[threadIdx.x*pointSize], nDim) == 1)
		flag = 0;

	__syncthreads();
    
    	//if (threadIdx.x==0) {
    	    eliminated[blockIdx.x] = flag;
	//}
}

/**
 * Insert the results and reorder into temp array in parallel.
 */ 
__global__ void insertResults(float *d_fr_iteration, float *temp, int *eliminated, int *scanoutput, int pointSize) {
	if (eliminated[blockIdx.x] == 1) {
		//insert the non-dominated points
		temp[(scanoutput[blockIdx.x]-1)*pointSize+threadIdx.x] = d_fr_iteration[blockIdx.x*pointSize+threadIdx.x];
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
 * Returns a sprimed & non-dominating front relative to point p at index.
 */
void limitset(int index, FRONT front) {
	int z = front.nPoints-1-index;
	
	/* <1> copy front to device memory */
	double h_front[front.nPoints*n];
	for (int i = 0; i < front.nPoints; i++) {
		for (int j = 0; j < n; j++) {
			h_front[i*n+j] = front.points[i].objectives[j];
		}
	}
	cudaMemcpy(d_front, h_front, front.nPoints*n*sizeof(double), cudaMemcpyHostToDevice);

	/* <2> Sprimes the front */
	//sprimeFront<<< z, n >>>( d_temp, d_front, index);
    	cudaThreadSynchronize(); // block until the device has completed

	/* <3> Compute eliminated array */
	//computeEliminatedArray<<< z, z >>>(d_temp, n, d_eliminated);
	cudaThreadSynchronize();
	
	int N = z;
	int blockSize = 512;
	int nBlocks = N/blockSize + (N%blockSize == 0 ? 0:1);

	/* <4> Compute prefix-sum in parallel */
	scan_best<<< nBlocks, blockSize/2, sizeof(int)*(blockSize) >>>(d_scanoutput, d_eliminated, blockSize);
	cudaThreadSynchronize();
	scan_inclusive<<< 1, z >>>(d_scanoutput, d_eliminated, z);  //make the result into an inclusive scan result.
	cudaThreadSynchronize();

	/* <5> Insert the results into temp buffer */
	//insertResults<<<z,n>>> (d_temp, d_temp2, d_eliminated, d_scanoutput);
	cudaThreadSynchronize();
	
	/* <6> Copy final results from device buffer back to host */
	cudaMemcpy(h_front, d_temp2, z*n*sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < z; i++) {
		for (int j = 0; j < n; j++) {
			fr[iteration].points[i].objectives[j] = h_front[i*n+j];
		}
	}

	/* <7> Update number of points */
	cudaMemcpy(&fr[iteration].nPoints, &d_scanoutput[z-1], sizeof(int), cudaMemcpyDeviceToHost); //update number of points
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
// Start of CUDA CODE
////////////////////////////////////////////////////////////////

/**
 * prints a front located on device
 */
void printfront(float *d_front, int numPoints) {
	printf("----------------------------------\n");
	float *front = (float *) malloc(frontSize*sizeof(float));
	cudaMemcpy(front, d_front, frontSize*sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < numPoints; i++) {
		for (int j = 0; j < n; j++) {
			printf("%1.1f ", front[i*pointSize+j]);
		}
		printf("\n");
	}
	printf("----------------------------------\n");
	free(front);
}

/**
 * Returns a sprimed & non-dominating front relative to point p at index.
 */
void limitset() {
	// TODO make this a kernel which calls many device functions 
	// TODO kernels may need to be passed the number of points in the front (from nPointsStack)
	// TODO &d_frontsArray[frontSize*iteration] may need to be changed to d_frontsArray+frontSize*iteration

	// sets the number of points in sprimed front
	int z = nPointsStack[iteration-1] - 1 - indexStack[iteration-1];

//printf("blah\n");
//printfront(&d_frontsArray[frontSize*(iteration-1)], nPointsStack[iteration-1]);
//printf("blah\n");
	// sprimes the front and store it into temporary storage
	sprimeFront<<< z, n >>>( d_temp, &d_frontsArray[frontSize*(iteration-1)], indexStack[iteration-1], pointSize);
    	cudaThreadSynchronize();

	// compute eliminated array and store it in d_eliminated
	computeEliminatedArray<<< z, z >>>(d_temp, n, d_eliminated, pointSize);
	cudaThreadSynchronize();
	
	// compute parallel prefix sum and store the result in d_scanoutput
	// TODO may need to make use of cudpp for this
	scan_best<<< 256, 512/2, sizeof(int)*(512) >>>(d_scanoutput, d_eliminated, 512);
	cudaThreadSynchronize();
	scan_inclusive<<< 1, z >>>(d_scanoutput, d_eliminated, z);  //make the result into an inclusive scan result.
	cudaThreadSynchronize();

	// compute the results and store it in frontArray
	insertResults<<<z,n>>> (d_temp, &d_frontsArray[frontSize*iteration], d_eliminated, d_scanoutput, pointSize);
	cudaThreadSynchronize();

	// update number of points to the host
	cudaMemcpy(&nPointsStack[iteration], &d_scanoutput[z-1], sizeof(int), cudaMemcpyDeviceToHost); //update number of points
//printfront(&d_frontsArray[frontSize*(iteration)], nPointsStack[iteration]);
}

/**
 * @param front front to sort
 * @param numElements number of points
 * @param size size of each point
 */
void parallelSort(float *d_in, int numElements) {

	// set the config
	CUDPPConfiguration config;
	//config.op = CUDPP_MAX;
	config.datatype = CUDPP_FLOAT;
	config.algorithm = CUDPP_SORT_RADIX_GLOBAL;
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
	
	// allocate array for sorted results
	float *d_out;
	cudaMalloc( (void **) &d_out, numElements*sizeof(float));

	// Run the sort 
	cudppSort(sortPlan, d_out, d_in, numElements);

	// TODO reassign pointers and remove costly memcpy operation
	//d_in = d_out;
	cudaMemcpy(d_in, d_out, numElements*sizeof(float), cudaMemcpyDeviceToDevice);

	// Destroy the plan
	result = cudppDestroyPlan(sortPlan);
	if (CUDPP_SUCCESS != result)
	{
		printf("Error destroying CUDPPPlan\n");
		exit(-1);
	}

	// TODO reuse config and destroy plan at the end
}

__global__ void initialise(int *prevOrder) {
	prevOrder[threadIdx.x] = threadIdx.x;
}

__global__ void initialiseKeys(float *keys, float *d_in, int i, int pointSize) {
	keys[threadIdx.x] = d_in[threadIdx.x*pointSize+i];
}

__global__ void initialiseUsed(int *used) {
	used[threadIdx.x] = -1;
}

__global__ void setuporder(int *used, float *d_in, int *prevOrder, float *keys, int numElements, int i, int pointSize, int *neworder) {
	for (int j = 0; j < numElements; j++)    {
		for (int k = 0; k < numElements; k++)    {
	   
			if (used[k] == -1 && d_in[prevOrder[k]*pointSize+i] == keys[j]){
				neworder[j] = prevOrder[k];
				used[k] = 0;
				break;
	    		}
		}
	}

}

__global__ void arrange(float *d_out, float *d_in, int *prevOrder, int pointSize) {

	// TODO this can be further parallelised
	// rearrange into a temporary array
	for (int i = 0; i < pointSize; i++) {
		d_out[threadIdx.x*pointSize+i] = d_in[prevOrder[threadIdx.x]*pointSize+i];
	}
	
}

void printElements(float *elements, int numElements) {
	float *sup = (float *) malloc(sizeof(float)*numElements);
	cudaMemcpy(sup, elements, sizeof(float)*numElements, cudaMemcpyDeviceToHost);
	for (int i = 0; i < numElements; i++) {
		printf("%f ", sup[i]);
	}
	printf("\n");
}

void printElements(int *elements, int numElements) {
	int *sup = (int *) malloc(sizeof(int)*numElements);
	cudaMemcpy(sup, elements, sizeof(int)*numElements, cudaMemcpyDeviceToHost);
	for (int i = 0; i < numElements; i++) {
		printf("%d ", sup[i]);
	}
	printf("\n");
}

void sortPoints(float *d_in, int numElements) {
	int *prevOrder;
	float *keys;
	int *neworder;
	int *used;

	cudaMalloc( (void **) &prevOrder, numElements*sizeof(int));
	initialise<<<1,numElements>>>(prevOrder);
	
	cudaMalloc( (void **) &keys, numElements*sizeof(float));
	cudaMalloc( (void **) &neworder, numElements*sizeof(int));
	cudaMalloc( (void **) &used, numElements*sizeof(int));

	for (int i = 0; i < n; i++) {
		initialiseKeys<<<1, numElements>>>(keys, d_in, i, pointSize);
	   	initialiseUsed<<<1, numElements>>>(used);

		parallelSort(keys, numElements);

		setuporder<<<1,1>>>(used, d_in, prevOrder, keys, numElements, i, pointSize, neworder);

		cudaMemcpy(prevOrder, neworder, sizeof(int)*numElements, cudaMemcpyDeviceToDevice);
        }

	// allocate array for arranged results
	float *d_out;
	cudaMalloc( (void **) &d_out, numElements*pointSize*sizeof(float));

	//arrange the order according to the last objectives
	arrange<<< 1, numElements>>>(d_out, d_in, prevOrder, pointSize);

	// copy d_out back to d_in
	cudaMemcpy(d_in, d_out, numElements*pointSize*sizeof(float), cudaMemcpyDeviceToDevice);
}


//////////////////////////////////////////////////////////
//  HV CUDA
//////////////////////////////////////////////////////////

__global__ void parallel_multiply(float *d_ehvStack, float *d_frontsArray, int iteration, int frontSize, int pointSize, int index, int n) {
	d_ehvStack[iteration] = 1;

	for (int i = 0; i < n; i++)  {
		d_ehvStack[iteration] *= d_frontsArray[frontSize*iteration+pointSize*index+i];
	}
}

__global__ void multiply2(float *d_hvStack, float *d_frontsArray, int iteration, int frontSize, int pointSize, int index, float *d_ehvStack, int n) {
	d_hvStack[iteration] = d_frontsArray[frontSize*iteration+pointSize*index+n] * d_ehvStack[iteration];
}

__global__ void compute2d(float *d_ehvStack, int iteration, float *d_frontsArray, int pointSize, int nPoints, int frontSize, float *d_hvStack, int index, int n ) {
	d_ehvStack[iteration] -= d_frontsArray[frontSize*(iteration+1)+pointSize*0+0] * d_frontsArray[frontSize*(iteration+1)+pointSize*0+1]; 
	for (int i = 1; i < nPoints; i++) {
		d_ehvStack[iteration] -= d_frontsArray[frontSize*(iteration+1)+pointSize*i+0] * 
					(d_frontsArray[frontSize*(iteration+1)+pointSize*i+1] - d_frontsArray[frontSize*(iteration+1)+pointSize*(i-1)+1]);
	}
	d_hvStack[iteration] += d_frontsArray[frontSize*iteration+pointSize*index+n] * d_ehvStack[iteration];
}

__global__ void computeFinishedLevel(float *d_ehvStack, float *d_hvStack, int iteration, float *d_frontsArray, int frontSize, int pointSize, int n, int index) {
	d_ehvStack[iteration] -= d_hvStack[iteration+1]; 
      	d_hvStack[iteration] += d_frontsArray[frontSize*iteration+pointSize*index+n] * d_ehvStack[iteration];
}

void hvparallel() {

	// sort the array
	sortPoints(&d_frontsArray[frontSize*0], nPointsStack[0]); // sorts the points located in front[0], use nPointsStack[0] for the number of points

	indexStack[0] = nPointsStack[0] - 1;

	// TODO host cannot access device memory ehv, and hv and frontsArray, need CUDA kernels for this
	// TODO d_frontsArray , d_hvStack and d_ehvStack is not possible
	while (indexStack[0] >= 0) {
		if (indexStack[iteration] < 0) {
			iteration--; 
			computeFinishedLevel<<<1,1>>>(d_ehvStack, d_hvStack, iteration, d_frontsArray, frontSize, pointSize, n, indexStack[iteration]);
       			indexStack[iteration]--;
       			n++;
		} else if (n == 2) {
			iteration--;
       			compute2d<<<1, 1 >>>(d_ehvStack, iteration, d_frontsArray, pointSize, nPointsStack[iteration+1], frontSize, d_hvStack, indexStack[iteration], n);
      			indexStack[iteration]--;
       			n++;
		} else {
      			n--;

       			parallel_multiply<<< 1, 1>>>(d_ehvStack, d_frontsArray, iteration, frontSize, pointSize, indexStack[iteration], n);
			cudaThreadSynchronize();

       			if (indexStack[iteration] == nPointsStack[iteration] - 1) {
				multiply2<<<1, 1>>>(d_hvStack, d_frontsArray, iteration, frontSize, pointSize, indexStack[iteration], d_ehvStack, n);
        			cudaThreadSynchronize();
          			indexStack[iteration]--;
          			n++;
			} else {
         			iteration++;
          			limitset(); 
          			sortPoints(&d_frontsArray[frontSize*iteration], nPointsStack[iteration]);
          			indexStack[iteration] = nPointsStack[iteration]-1;
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
	frontSize = maxPoints*maxDimensions;
	pointSize = maxDimensions;
	cudaMalloc((void **) &d_frontsArray, frontSize * maxDimensions * sizeof(float));
	cudaMalloc((void **) &d_ehvStack, sizeof(float) * maxDimensions);
	cudaMalloc((void **) &d_hvStack, sizeof(float) * maxDimensions);

	// allocate cpu memory Stacks
	indexStack = (int *) malloc(sizeof(int) * maxDimensions);
	nPointsStack = (int *) malloc(sizeof(int) * maxDimensions);

	// process each front to get the hypervolumes
	for (int i = 0; i < f->nFronts; i++) {
		// read each front
		FRONT front = f->fronts[i];
		n = front.n;
		nPointsStack[0] = front.nPoints;

		// CHECK UNIQUE NESS OF OBJECTIVES
		/*for (int x = n-1; x >= 0; x--) {
			for (int j = 0; j < front.nPoints; j++) {
				for (int k = 0; k < front.nPoints; k++) {
					if (k == j) continue; //avoid checking against itself

					if (front.points[j].objectives[x] == front.points[k].objectives[x]) {
						fprintf(stderr, "data set are not unique in every objectives\n");
						printf("error!!!\n");
						exit(EXIT_FAILURE);
					}
				}
			}
		}*/

		// copy front to device memory
		float h_front[front.nPoints*pointSize]; 
		for (int j = 0; j < front.nPoints; j++) {
			for (int k = 0; k < n; k++) {
				h_front[j*pointSize+k] = front.points[j].objectives[k];
			}
		}
		cudaMemcpy(d_frontsArray, h_front, frontSize*sizeof(float), cudaMemcpyHostToDevice);		

		// run hv parallel
		hvparallel();

 		// copy back hvresult
		float *hvResult = (float *) malloc(sizeof(float));
		cudaMemcpy(hvResult, d_hvStack, sizeof(float), cudaMemcpyDeviceToHost);

		// print them out
		printf("Calculating Hypervolume for Front:%d...\n", i+1);
		printf("\t\t\t\t\t%f\n", hvResult[0]);
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

