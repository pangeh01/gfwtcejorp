#include "read.c"
#include <stdbool.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <math.h>

int maxDepth = -1;	//the maximum depth you have reached
int maxDimensions = 0;	//the max number of dimensions in the fronts
int n = 0; //the dimension of the current front we are working on
int maxPoints = 0;  //the max number of points in the fronts
int iteration = 0;	//depth of the recursion starting from 0
extern double ehv(int index, FRONT front);
extern double hypervolume(FRONT front);
FRONT *fr;	//storage for storing array of sprimed/non-dominating fronts as we go deeper into the recursion

/* returns a minimum of two numbers */
#define MIN(x, y) ((x < y) ? (x) : (y))

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

/* returns a sprimed & non-dominating front relative to point p at index */
void limitset(int index, FRONT front) {
	
	//allocate only if it's a new depth in the iteration
	/*if (iteration > maxDepth) {
		maxDepth = iteration;
		//fr[iteration].n = nDimensions;
		fr[iteration].points = malloc(sizeof(POINT) * maxPoints);
		for (int j = 0; j < maxPoints; j++) {
			//only need to malloc for n (number of Dimensions)
			fr[iteration].points[j].objectives = malloc(sizeof(OBJECTIVE) * n);
		}
	}*/
	
	int z = front.nPoints-1-index;
	//sprimes the front and insert it into allocated array of fronts
	for (int i = 0; i < z; i++) {
		for (int j = 0; j < n; j++) {
			fr[iteration].points[i].objectives[j] = MIN(front.points[index].objectives[j], front.points[i+1+index].objectives[j]);
		}
	}

	//make the sprimed front into a non-dominating front
	POINT temp;					//temporary buffer
	fr[iteration].nPoints = 1;	//non-dominating storage set initially have at least 1 point (the first point is stored)

	//loop through each point in the sprimed set, starting from the second point
	//[i] refer to the points in sprimed set, [j] refer to the storaged (non-dominating) set 
	bool add;
	int j;
	for (int i = 1; i < z; i++) {
		add = true;		//whether to store the point in memory
		j = 0;
		while (j < fr[iteration].nPoints && add) {
			switch (dominated(fr[iteration].points[i], fr[iteration].points[j])) {
				case -1:
					//if point a dominates b, discard point b
					temp = fr[iteration].points[fr[iteration].nPoints - 1];
					fr[iteration].points[fr[iteration].nPoints - 1] = fr[iteration].points[j];
					fr[iteration].points[j] = temp;
					fr[iteration].nPoints--;
					break;
				case 1: case 2:
					//if b dominates a (1), or a == b (2), point a is not in non-dominating set (stop checking)
					add = false;
					break;
				default:
					j++;
			}
		}
		
		//if a is non-dominating point addition to the storaged set, add the point in storage
		if (add) {
			temp = fr[iteration].points[fr[iteration].nPoints];
			fr[iteration].points[fr[iteration].nPoints] = fr[iteration].points[i]; 
			fr[iteration].points[i] = temp;
			fr[iteration].nPoints++;
		}
    }
	
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
		iteration++;
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
	for (int i = 0; i < front.nPoints; i++) {
		sumhv += front.points[i].objectives[n] * ehv(i, front);
	}
	n++;
	
	return sumhv;
}

int main(int argc, char *argv[]) {
	
	FILECONTENTS *f = readFile(argv[1]);
	
	struct timeval tv1, tv2;
	struct rusage ru_before, ru_after;
	getrusage (RUSAGE_SELF, &ru_before);
	
	//find the max number of Points, and the max number of Dimensions
	for (int i = 0; i < f->nFronts; i++) {
		if (f->fronts[i].nPoints > maxPoints) 
			maxPoints = f->fronts[i].nPoints;
		if (f->fronts[i].n > maxDimensions) 
			maxDimensions = f->fronts[i].n;
    }
	
	//allocate an array of fronts for nDimensions
	fr = malloc(sizeof(FRONT) * maxDimensions);
	for (int i = 0; i < maxDimensions; i++) {
		fr[i].points = malloc(sizeof(POINT) * maxPoints);
		for (int j = 0; j < maxPoints; j++) {
		//only need to malloc for n (number of Dimensions)
			fr[i].points[j].objectives = malloc(sizeof(OBJECTIVE) * maxDimensions);
		}
	}
	
	//process each front to get the hypervolumes
	for (int i = 0; i < f->nFronts; i++) {
		n = f->fronts[i].n;
		//iteration = 0;							//initialise iteration
		printf("Calculating Hypervolume for Front:%d...\n", i+1);
		printf("\t\t\t\t\t%1.10f\n", hypervolume(f->fronts[i]));
		
	}
	
	getrusage (RUSAGE_SELF, &ru_after);
	tv1 = ru_before.ru_utime;
	tv2 = ru_after.ru_utime;
	printf("Average time = %fs\n", (tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6) / f->nFronts);
	
	//free the storage
	for (int i = 0; i < maxDimensions+1; i++) {
		for (int j = 0; j < maxPoints; j++) {
			free(fr[i].points[j].objectives);
		}
		free(fr[i].points);
	}
	free(fr);
	
	return 0;
}