
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
extern void printElements(float *elements, int numElements);
extern void printElements(int *elements, int numElements);

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

// for sorting
int *order;
float *keys;

CUDPPHandle sortPlan;
CUDPPHandle scanPlan;

//NOte: n is needed for slicing and sorting, iteration is needed for saving array of fronts when going deeper into recursion

unsigned int timer = 0;

void create() {
timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
}

/**
 *  Timer Functions
 */
void start()
{
    /////////////////////////////////////////////////////////////////////
    // Create and start a timer called "timer"
    // alls to create ans start times are enveloped in the CUT_SAFE_CALL
    // This CUDA Utility Tool checks for errors upon return.
    // If an error is found, it prints out and error message, file name,
    // and line number in file where the error can be found
    /////////////////////////////////////////////////////////////////////
    CUT_SAFE_CALL(cutStartTimer(timer));
}

void stop() {
 // Stop the timer
    CUT_SAFE_CALL(cutStopTimer(timer));
}

void end() {
printf( "Processing time: %f (ms)\n", cutGetTimerValue(timer));

    // Delete the timer
    CUT_SAFE_CALL(cutDeleteTimer(timer));
}
/////////////////////////////////////////////////////////
// GPU Kernel Functions
/////////////////////////////////////////////////////////

/**
 * Sprimes a front in parallel.
 */
__global__ void sprimeFront(float *result, float *frontPoints_device, int index, int pointSize) {
	extern __shared__ float sdata2[];

	sdata2[blockIdx.x*pointSize+threadIdx.x] = MIN(frontPoints_device[index*pointSize+threadIdx.x], 
									frontPoints_device[(blockIdx.x+1+index)*pointSize+threadIdx.x]);

	result[blockIdx.x*pointSize+threadIdx.x] = sdata2[blockIdx.x*pointSize+threadIdx.x];
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
    
    	eliminated[blockIdx.x] = flag;
}

/**
 * Insert the results and reorder into temp array in parallel.
 */ 
__global__ void insertResults(float *d_fr_iteration, int *eliminated, int *scanoutput, int pointSize) {
	if (eliminated[blockIdx.x] == 1) {
		//insert the non-dominated points
		d_fr_iteration[(scanoutput[blockIdx.x]-1)*pointSize+threadIdx.x] = d_fr_iteration[blockIdx.x*pointSize+threadIdx.x];
	}
}


////////////////////////////////////////////////////////////////
// Start of CUDA CODE
////////////////////////////////////////////////////////////////

/**
 * Returns a sprimed & non-dominating front relative to point p at index.
 */
void limitset() {
	// sets the number of points in sprimed front
	int z = nPointsStack[iteration-1] - 1 - indexStack[iteration-1];

	// sprimes the front and store it into temporary storage
	sprimeFront<<< z, n, pointSize*z*sizeof(float) >>>(&d_frontsArray[frontSize*(iteration)], &d_frontsArray[frontSize*(iteration-1)], indexStack[iteration-1], pointSize);
    	//cudaThreadSynchronize();

	// compute eliminated array and store it in d_eliminated
	computeEliminatedArray<<< z, z >>>(&d_frontsArray[frontSize*(iteration)], n, d_eliminated, pointSize);
	//cudaThreadSynchronize();
	
	// Run the scan
	cudppScan(scanPlan, d_scanoutput, d_eliminated, z);
	//cudaThreadSynchronize();

	// compute the results and store it in frontArray
	insertResults<<<z,n>>> (&d_frontsArray[frontSize*iteration], d_eliminated, d_scanoutput, pointSize);
	//cudaThreadSynchronize();

	// update number of points to the host
	cudaMemcpy(&nPointsStack[iteration], &d_scanoutput[z-1], sizeof(int), cudaMemcpyDeviceToHost); //update number of points
}

void setUpPlan() {
	// set the config
	CUDPPConfiguration config;
	//config.op = CUDPP_MAX;
	config.datatype = CUDPP_FLOAT;
	config.algorithm = CUDPP_SORT_RADIX_GLOBAL;
	config.options = CUDPP_OPTION_FORWARD;
	
	// create the plan
	cudppPlan(&sortPlan, config, frontSize/pointSize, 1, 0);

    	//config.op = CUDPP_ADD;
    	config.datatype = CUDPP_INT;
    	config.algorithm = CUDPP_SCAN;
   	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
    
   	CUDPPResult result = cudppPlan(&scanPlan, config, frontSize/pointSize, 1, 0); 
}

__global__ void initialise(int *prevOrder) {
	prevOrder[threadIdx.x] = threadIdx.x;
}

__global__ void initialiseKeys(float *keys, float *d_in, int i, int pointSize) {
	extern __shared__ float sharedKeys[];

	sharedKeys[threadIdx.x] = d_in[threadIdx.x*pointSize+i];

	keys[threadIdx.x] = sharedKeys[threadIdx.x];
}

__global__ void setuporder(float *d_in, int *order, float *keys, int i, int pointSize) {
	extern __shared__ int sharedData[];

	sharedData[threadIdx.x+blockDim.x] = -1;

	for (int j = 0; j < blockDim.x; j++)    {
		for (int k = 0; k < blockDim.x; k++)    {
	 
			if (sharedData[k+blockDim.x] == -1 && d_in[order[k]*pointSize+i] == keys[j]){
				sharedData[j] = order[k];
				sharedData[k+blockDim.x] = 0;
				break;
	    		}
		}
	}

	order[threadIdx.x] = sharedData[threadIdx.x];
}

__global__ void arrange(float *d_out, float *d_in, int *prevOrder) {
	extern __shared__ float tempArray[];

	tempArray[threadIdx.x*gridDim.x+blockIdx.x] = d_in[prevOrder[threadIdx.x]*gridDim.x+blockIdx.x];
	
	d_out[threadIdx.x*gridDim.x+blockIdx.x] = tempArray[threadIdx.x*gridDim.x+blockIdx.x];
}

void sortPoints(float *d_in, int numElements) {

	initialise<<<1,numElements>>>(order);

	for (int i = 0; i < n; i++) {
		initialiseKeys<<<1, numElements, numElements*sizeof(int)>>>(keys, d_in, i, pointSize);

		cudppSort(sortPlan, keys, keys, numElements);

		setuporder<<<1,numElements, numElements*sizeof(int)*2>>>(d_in, order, keys, i, pointSize);
        }

	//arrange the order according to the last objectives
	arrange<<< pointSize, numElements, pointSize*numElements*sizeof(float)>>>(d_in, d_in, order);

}


//////////////////////////////////////////////////////////
//  HV CUDA
//////////////////////////////////////////////////////////

__global__ void reduce0(float *g_idata, float *g_odata){ 
	extern __shared__ float sdata[]; 
	// each thread loadsone element from global to shared mem 
	unsigned int tid = threadIdx.x; 
	unsigned int i= blockIdx.x*blockDim.x+ threadIdx.x; 
	sdata[tid] = g_idata[i]; 
	__syncthreads(); 

	// do reduction in shared mem 
	for(unsigned int s=1; s < blockDim.x; s *= 2) { 
		if(tid % (2*s) == 0){ 
			sdata[tid] *= sdata[tid + s]; 
		} 
		__syncthreads(); 
	} 
	// write result for this block to global mem 
	if(tid == 0) g_odata[blockIdx.x] = sdata[0]; 
}

__global__ void parallel_multiply(float *d_ehvStack, float *d_frontsArray, int n) {
	d_ehvStack[0] = 1;

	for (int i = 0; i < n; i++)  {
		d_ehvStack[0] *= d_frontsArray[i];
	}
}

__global__ void multiply2(float *d_hvStack, float *d_frontsArray, float *d_ehvStack) {
	d_hvStack[0] = d_frontsArray[0] * d_ehvStack[0];
}

__global__ void compute2d(float *d_ehvStack, int iteration, float *d_frontsArray, int pointSize, int nPoints, int frontSize, float *d_hvStack, int index, int n ) {
	d_ehvStack[iteration] -= d_frontsArray[frontSize*(iteration+1)] * d_frontsArray[frontSize*(iteration+1)+1]; 
	for (int i = 1; i < nPoints; i++) {
		d_ehvStack[iteration] -= d_frontsArray[frontSize*(iteration+1)+pointSize*i] * 
					(d_frontsArray[frontSize*(iteration+1)+pointSize*i+1] - d_frontsArray[frontSize*(iteration+1)+pointSize*(i-1)+1]);
	}
	d_hvStack[iteration] += d_frontsArray[frontSize*iteration+pointSize*index+n] * d_ehvStack[iteration];
}

__global__ void computeFinishedLevel(float *d_ehvStack, float *d_hvStack, int iteration, float *d_frontsArray, int frontSize, int pointSize, int n, int index) {
	d_ehvStack[iteration] -= d_hvStack[iteration+1]; 
      	d_hvStack[iteration] += d_frontsArray[frontSize*iteration+pointSize*index+n] * d_ehvStack[iteration];
}

void hvparallel() {  
	setUpPlan();

	// sort the array
	sortPoints(&d_frontsArray[0], nPointsStack[0]); // sorts the points located in front[0], use nPointsStack[0] for the number of points
create();
	indexStack[0] = nPointsStack[0] - 1;

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

			int pointIdx = frontSize*iteration+pointSize*indexStack[iteration];
       			parallel_multiply<<< 1, 1>>>(&d_ehvStack[iteration], &d_frontsArray[pointIdx], n);
			
       			if (indexStack[iteration] == nPointsStack[iteration] - 1) {
				int dimIdx = frontSize*iteration+pointSize*indexStack[iteration]+n;
				multiply2<<<1, 1>>>(&d_hvStack[iteration], &d_frontsArray[dimIdx], &d_ehvStack[iteration]);
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
end();
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
	
	// find the max number of Points, and the max number of Dimensions
	int maxDimensions = 0;	//the max number of dimensions in the fronts
	int maxPoints = 0;  //the max number of points in the fronts
	for (int i = 0; i < f->nFronts; i++) {
		if (f->fronts[i].nPoints > maxPoints) 
			maxPoints = f->fronts[i].nPoints;
		if (f->fronts[i].n > maxDimensions) 
			maxDimensions = f->fronts[i].n;
  	}

	// allocate memory for limitset
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

	// allocate cuda memory
	cudaMalloc( (void **) &order, maxPoints*sizeof(int));
	cudaMalloc( (void **) &keys, maxPoints*sizeof(float));

	// allocate cpu memory Stacks
	indexStack = (int *) malloc(sizeof(int) * maxDimensions);
	nPointsStack = (int *) malloc(sizeof(int) * maxDimensions);

	// process each front to get the hypervolumes
	for (int i = 0; i < f->nFronts; i++) {
		// read each front
		FRONT front = f->fronts[i];
		n = front.n;
		nPointsStack[0] = front.nPoints;

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

//////////////////////////////////
// HELPER METHODS
//////////////////////////////////

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

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(-1);
    }                         
}

