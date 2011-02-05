#include "read.c"
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>

//int maxDepth = -1;	//the maximum depth you have reached
int n = 0; //the dimension of the current front we are working on
int iteration = 0;	//depth of the recursion starting from 0
//extern double ehv(int index, FRONT front);
//extern double hypervolume(FRONT front);
double hypervolume(FRONT);
FRONT *fr;	//storage for storing array of sprimed/non-dominating fronts as we go deeper into the recursion
#define MIN(x, y) ((x < y) ? (x) : (y))

/*__device__ int min(int x, int y) {
	if (x < y) return x;
	return y;
}*/

/* prints the front */
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

/* my kernel functions */
__global__ void sprimeFront(double *frPoints_device, double *frontPoints_device, int index) {
	frPoints_device[blockIdx.x*blockDim.x+threadIdx.x] = MIN(frontPoints_device[index*blockDim.x+threadIdx.x], frontPoints_device[(blockIdx.x+1+index)*blockDim.x+threadIdx.x]);
}

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

/*__device__ int contains(int *results, int dimension, int number) {
	for (int i = 0; i < dimension; i++) {
		if (results[i] == number) return 1;
	}
	return 0;
}

__global__ void computeEliminatedArray(double *d_fr_iteration, int nDim, int *eliminated) {
	extern __shared__ int dominatedresults[];
	
	dominatedresults[threadIdx.x] = dominated(d_fr_iteration+blockIdx.x*nDim , d_fr_iteration+threadIdx.x*nDim, nDim);

	__syncthreads();
	
	if (contains(dominatedresults, blockDim.x, 1)) { // && eliminated[blockIdx.x] == -1) {
		eliminated[blockIdx.x] = 1;
	} else {
		eliminated[blockIdx.x] = 0;
	}

}*/

/* computes eliminated array (known bug: equal points will not be eliminated) */
__global__ void computeEliminatedArray(double *d_fr_iteration, int nDim, int *eliminated, int *equality) {
    	__shared__ int flag;
	//__shared__ int[] equality;

    	flag = 0;
    	__syncthreads();
	
	int result = dominated(d_fr_iteration+blockIdx.x*nDim , d_fr_iteration+threadIdx.x*nDim, nDim);
    	if (result == 1)
		flag = 1;
	/*if (result == 2) {
		if (blockIdx.x == threadIdx.x) {
		equality[threadIdx.x] = 1;
	}*/

	__syncthreads();
    
    	if (threadIdx.x==0)
    	    eliminated[blockIdx.x] = flag;
}

//not working properly because d_count not counting properly (due to parallelism)
__global__ void insertResults(double *d_fr_iteration, double *temp, int *eliminated, int *d_count) {
	if (eliminated[blockIdx.x] == 0) {
		temp[*d_count*blockDim.x+threadIdx.x] = d_fr_iteration[blockIdx.x*blockDim.x+threadIdx.x];
		(*d_count)++;
	} else {
		temp[(gridDim.x-1-(blockIdx.x-*d_count))*blockDim.x+threadIdx.x] = d_fr_iteration[blockIdx.x*blockDim.x+threadIdx.x];
	}
}
/* my kernel functions finished */

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}

/* returns a sprimed & non-dominating front relative to point p at index */
void limitset(int index, FRONT front) {
	int z = front.nPoints-1-index;
	
	//temporary device buffer
	double *d_fr_iteration;
	cudaMalloc( (void **) &d_fr_iteration, z*n*sizeof(double));

	//copy front into device memory
	double *d_front;
	cudaMalloc( (void **) &d_front, front.nPoints*n*sizeof(double));
	for (int i = 0; i < front.nPoints; i++) {
		cudaMemcpy(d_front+i*n, front.points[i].objectives, n*sizeof(double), cudaMemcpyHostToDevice);
	}
	
	//sprimes the front
	sprimeFront<<< z, n >>>( d_fr_iteration, d_front, index);
	
	// block until the device has completed
    	cudaThreadSynchronize();

    	// check if kernel execution generated an error
    	checkCUDAError("spriming execution");
	
	int *d_eliminated;
	cudaMalloc((void **) &d_eliminated, z*sizeof(int));
	computeEliminatedArray<<< z, z >>>(d_fr_iteration, n, d_eliminated);
	
	cudaThreadSynchronize();

	//kernel code (not working)
	/*int *count;
	int h_count = 0;
	cudaMalloc((void **) &count, sizeof(int));
	cudaMemcpy(count, &h_count, sizeof(int), cudaMemcpyHostToDevice);
	
	double *temp;
	cudaMalloc((void**) &temp, sizeof(double)*z*n);
	insertResults<<<z,n>>> (d_fr_iteration, temp, d_eliminated, count);
	cudaThreadSynchronize();
	
	//copy device buffer back to host
	cudaMemcpy(d_fr_iteration, temp, z*n*sizeof(double), cudaMemcpyDeviceToDevice);
	for (int i = 0; i < z; i++) {
		cudaMemcpy(fr[iteration].points[i].objectives, temp+i*n, n*sizeof(double), cudaMemcpyDeviceToHost);
	}
	cudaMemcpy(&fr[iteration].nPoints, count, sizeof(int), cudaMemcpyDeviceToHost);

	// Check for any CUDA errors
    	checkCUDAError("cudaMemcpy back to host");*/
	
	//use host code instead
	for (int i = 0; i < z; i++) {
		cudaMemcpy(fr[iteration].points[i].objectives, d_fr_iteration+i*n, n*sizeof(double), cudaMemcpyDeviceToHost);
	}
	int *h_eliminated = (int *) malloc(z*sizeof(int));
	cudaMemcpy(h_eliminated, d_eliminated, z*sizeof(int), cudaMemcpyDeviceToHost);
	
	fr[iteration].nPoints = 0;
	POINT temp;
	for (int i = 0; i < z; i++) {
		if (h_eliminated[i] == 0) {
			temp = fr[iteration].points[fr[iteration].nPoints];
			fr[iteration].points[fr[iteration].nPoints] = fr[iteration].points[i];
			fr[iteration].points[i] = temp;
			fr[iteration].nPoints++;
		}
	}

	//free device memory
	cudaFree(d_fr_iteration);
	//cudaFree(temp);
	cudaFree(d_front);
	//cudaFree(count);
	cudaFree(d_eliminated);
	free(h_eliminated);
	
	printf("%d\n", fr[iteration].nPoints);
	printfront(fr[iteration]);
}

/* compare function for qsort sorting front in the last objective */
int compare (const void *a, const void *b)
{
	//n == maxDimensions-iteration
	for (int i = n - 1; i >= 0; i--) {
		if (((*(POINT *)a).objectives[i] > (*(POINT *)b).objectives[i])) return 1;
		if (((*(POINT *)a).objectives[i] < (*(POINT *)b).objectives[i])) return -1;
	}
	return 0;
}

/* returns the size of exclusive hypervolume of point p at index relative to a front set */
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

/* returns the size of hypervolume a front) */
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
	
	return 0;
}
