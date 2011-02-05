
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

#include "read.c"
#include "scan_best_kernel.cu"
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>

//extern double ehv(int index, FRONT front);
//extern double hypervolume(FRONT front);

#define MIN(x, y) ((x < y) ? (x) : (y))

/////////////////////////////////////////////////////////
// Global Variables
/////////////////////////////////////////////////////////

//int maxDepth = -1;	//the maximum depth you have reached
int n = 0; //the dimension of the current front we are working on
int iteration = 0;	//depth of the recursion starting from 0
double hypervolume(FRONT);
FRONT *fr;	//storage for storing array of sprimed/non-dominating fronts as we go deeper into the recursion

/* Device global variables */
double *d_temp;
double *d_front;
int *d_eliminated;
int *d_scanoutput;
double *d_temp2;

/////////////////////////////////////////////////////////
// GPU Kernel Functions
/////////////////////////////////////////////////////////

/**
 * Sprimes a front in parallel.
 */
/*__global__ void sprimeFront(double *frPoints_device, double *frontPoints_device, int index) {
	frPoints_device[blockIdx.x*blockDim.x+threadIdx.x] = MIN(frontPoints_device[index*blockDim.x+threadIdx.x], 
	frontPoints_device[(blockIdx.x+1+index)*blockDim.x+threadIdx.x]);
}*/

/**
 * Device Function: Determine domination status of point A and B.
 * Similar to CPU implementation.
 *
 * returns 1 if point b dominates a
 * zero if non-dominating
 * returns -1 if point a dominates b
 * returns 2 if point is equal
 */
__device__ int dominated(double *point_a, double *point_b, int nDim) {
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
__device__ void computeEliminatedArray(double *d_fr_iteration, int nDim, int *eliminated, int z) {
    	__shared__ int flag;

    	flag = 1;
	__syncthreads();

	if (threadIdx.x < z && blockIdx.x < z) {
	if (dominated(&d_fr_iteration[blockIdx.x*nDim] , &d_fr_iteration[threadIdx.x*nDim], nDim) == 1)
		flag = 0;
	}
	__syncthreads();
    
    	//if (threadIdx.x==0) {
    	    eliminated[blockIdx.x] = flag;
	//}
}

/**
 * Insert the results and reorder into temp array in parallel.
 */ 
__global__ void insertResults(double *d_fr_iteration, double *temp, int *eliminated, int *scanoutput) {
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

/**
 * Sprimes a front in parallel.
 */
__device__ void sprimeFront(double *frPoints_device, double *frontPoints_device, int index, int nDim) {
	if (threadIdx.x < nDim) {
		frPoints_device[blockIdx.x*nDim+threadIdx.x] = MIN(frontPoints_device[index*nDim+threadIdx.x], 
		frontPoints_device[(blockIdx.x+1+index)*nDim+threadIdx.x]);
	}
}

__device__ void hello(double *frPoints_device, double *frontPoints_device, int index, int nDim, int *d_eliminated, int z ) {
	__shared__ int flag;

    	flag = 1;
	__syncthreads();

	if (dominated(&frPoints_device[blockIdx.x*nDim] , &frPoints_device[threadIdx.x*nDim], nDim) == 1)
		flag = 0;
	
	__syncthreads();
    
	
    	d_eliminated[blockIdx.x] = flag;
}

__global__ void kernel(double *frPoints_device, double *frontPoints_device, int index, int nDim, int *d_eliminated, int z ) {
	sprimeFront(frPoints_device, frontPoints_device, index, nDim );
	//computeEliminatedArray(frPoints_device, nDim, d_eliminated, z );

	if (threadIdx.x < z && blockIdx.x < z) {
		hello(frPoints_device, frontPoints_device, index, nDim, d_eliminated, z);
	}
	//scan_best( );
	//scan_inclusive( );
	//insertResults( );
}

/*__global__ void kernel1(double *frPoints_device, double *frontPoints_device, int index, int nDim, int *d_eliminated, int z ) {
	if (threadIdx.x < z && blockIdx.x < z) {
		hello(frPoints_device, frontPoints_device, index, nDim, d_eliminated, z);
	}
}*/
/////////////////////////////////////////////////////////
// Helper methods
/////////////////////////////////////////////////////////

/**
 * Check if kernel execution generated an error.
 * Report the CUDA error.
 * Run the method after each kernel execution.
 */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(-1);
    }                         
}

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

	/* kernel code */
	kernel<<< 1000, 256>>>(d_temp, d_front, index, n, d_eliminated, z);	

	/* <2> Sprimes the front */
	//sprimeFront<<< 1000, 256 >>>( d_temp, d_front, index, n);
    	cudaThreadSynchronize(); // block until the device has completed
	//kernel1<<< 1000, 256>>>(d_temp, d_front, index, n, d_eliminated, z);
	/* <3> Compute eliminated array */
	//computeEliminatedArray<<< z, z >>>(d_temp, n, d_eliminated);
	cudaThreadSynchronize();
	
	/* <4> Compute prefix-sum in parallel */
	scan_best<<< 256, 512/2, sizeof(int)*(512) >>>(d_scanoutput, d_eliminated, 512);
	cudaThreadSynchronize();
	scan_inclusive<<< 1, z >>>(d_scanoutput, d_eliminated, z);  //make the result into an inclusive scan result.
	cudaThreadSynchronize();

	/* <5> Insert the results into temp buffer */
	insertResults<<<z,n>>> (d_temp, d_temp2, d_eliminated, d_scanoutput);
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
 * Compare function for qsort sorting front in the last objective.
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
double ehv(int index, FRONT front) {
	
	//hypervolume of a single poinit
	double ehv = 1;
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
double hypervolume(FRONT front) {
	//sort the front with qsort
	qsort(front.points, front.nPoints, sizeof (POINT), compare);
	
	//calculate for base case = 2D
	if (n==2) {
		double vol2d = (front.points[0].objectives[0] * front.points[0].objectives[1]);
		for (int i = 1; i < front.nPoints; i++) {
			vol2d += (front.points[i].objectives[0]) * 
						   (front.points[i].objectives[1] - front.points[i - 1].objectives[1]);
		}
		return vol2d;
	}
	
	double sumhv = 0;
	n--;
	//sum all the segments
	for (int i = front.nPoints - 1; i >= 0; i--)
		//for (int i = 0; i < front.nPoints; i++) //annoying bug that cause inaccurate results
		sumhv += front.points[i].objectives[n] * ehv(i, front);
	n++;
	
	return sumhv;
}

/////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
	
	FILECONTENTS *f = readFile(argv[1]);
	
	struct timeval tv1, tv2;
	struct rusage ru_before, ru_after;
	getrusage (RUSAGE_SELF, &ru_before);
	
	int maxDimensions = 0;	//the max number of dimensions in the fronts
	int maxPoints = 0;  //the max number of points in the fronts
	
	//find the max number of Points, and the max number of Dimensions
	for (int i = 0; i < f->nFronts; i++) {
		if (f->fronts[i].nPoints > maxPoints) 
			maxPoints = f->fronts[i].nPoints;
		if (f->fronts[i].n > maxDimensions) 
			maxDimensions = f->fronts[i].n;
  	}
	
	int maxd = maxDimensions-2;
	//allocate an array of fronts for nDimensions
	fr = (FRONT *) malloc(sizeof(FRONT) * maxd);
	for (int i = 0; i < maxd; i++) {
		fr[i].points = (POINT *) malloc(sizeof(POINT) * maxPoints);
		for (int j = 0; j < maxPoints; j++) {
		//only need to malloc for n (number of Dimensions)
			fr[i].points[j].objectives = (OBJECTIVE *) malloc(sizeof(OBJECTIVE) * (maxDimensions - (i+1)));
		}
	}
	
	/* allocate for cuda memory */
	cudaMalloc( (void **) &d_temp, maxPoints*maxDimensions*sizeof(double));
	cudaMalloc((void **) &d_scanoutput, (512)*sizeof(int));
	cudaMalloc((void **) &d_eliminated, (512)*sizeof(int));
	cudaMalloc( (void **) &d_front, maxPoints*maxDimensions*sizeof(double));
	cudaMalloc((void**) &d_temp2, sizeof(double)*maxPoints*maxDimensions);

	//process each front to get the hypervolumes
	for (int i = 0; i < f->nFronts; i++) {
		n = f->fronts[i].n;
		printf("Calculating Hypervolume for Front:%d...\n", i+1);
		printf("\t\t\t\t\t%1.10f\n", hypervolume(f->fronts[i]));
	}
	
	getrusage (RUSAGE_SELF, &ru_after);
	tv1 = ru_before.ru_utime;
	tv2 = ru_after.ru_utime;
	printf("Average time = %fs\n", (tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6) / f->nFronts);
	
	//free the storage
	for (int i = 0; i < maxd; i++) {
		for (int j = 0; j < maxPoints; j++) {
			free(fr[i].points[j].objectives);
		}
		free(fr[i].points);
	}
	free(fr);

	/* free cuda memory */
	cudaFree(d_temp);
	cudaFree(d_temp2);
	cudaFree(d_scanoutput);
	cudaFree(d_front);
	cudaFree(d_eliminated);

	return 0;
}
