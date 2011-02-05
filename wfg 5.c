#include "read.c"
#include <stdbool.h>

int maxDepth = -1;	//the maximum depth you have reached
int nDimensions = 0;	//the max number of dimensions in the fronts
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
		for (int k = 0; k < front.n; k++)
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
	for (int i = 0; i < nDimensions; i++) {
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
	if (iteration > maxDepth) {
		maxDepth = iteration;
		fr[iteration].n = nDimensions;
		fr[iteration].points = malloc(sizeof(POINT) * maxPoints);
		for (int j = 0; j < maxPoints; j++) {
			fr[iteration].points[j].objectives = malloc(sizeof(OBJECTIVE) * nDimensions);
		}
	}
	
	//sprimes the front and insert it into allocated array of fronts
	for (int i = 0; i < front.nPoints-1-index; i++) {
		for (int j = 0; j < front.n; j++) {
			fr[iteration].points[i].objectives[j] = MIN(front.points[index].objectives[j], front.points[i+1+index].objectives[j]);
		}
	}

	//make the sprimed front into a non-dominating front
	POINT temp;					//temporary buffer
	fr[iteration].nPoints = 1;	//non-dominating storage set initially have at least 1 point (the first point is stored)

	//loop through each point in the sprimed set, starting from the second point
	//[i] refer to the points in sprimed set, [j] refer to the storaged (non-dominating) set 
	for (int i = 1; i < front.nPoints-1-index; i++) {
		bool store = true;		//whether to store the point in memory

		for (int j =0; j < fr[iteration].nPoints && store; j++) {
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
					store = false;
					break;
			}
		}
		
		if (store) {
			//store the point a to the storaged set
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
	if (((*(POINT *)a).objectives[nDimensions-iteration-1] - (*(POINT *)b).objectives[nDimensions-iteration-1]) > 0)
		return 1;
	else
		return -1;
}

/* returns the size of exclusive hypervolume of point p at index relative to a front set */
double ehv(int index, FRONT front) {
	
	//hypervolume of a single poinit
	double ehv = 1;
	for (int i = 0; i < front.n; i++) {
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
	
	double sumhv = 0;
	
	//sort the front with qsort
	qsort(front.points, front.nPoints, sizeof (POINT), compare);
	
	//sum all the segments
	for (int i = 0; i < front.nPoints; i++) {
		sumhv += ehv(i, front);
	}
	
	return sumhv;
}

int main(int argc, char *argv[]) {
	
	FILECONTENTS *f = readFile(argv[1]);
	
	//find the max number of Points, and the max number of Dimensions
	for (int i = 0; i < f->nFronts; i++) {
		if (f->fronts[i].nPoints > maxPoints) 
			maxPoints = f->fronts[i].nPoints;
		if (f->fronts[i].n > nDimensions) 
			nDimensions = f->fronts[i].n;
    }
	
	//allocate an array of fronts for nDimensions
	//is it maxed at nDimensions or nPoints? nDimensions seem to work.
	fr = malloc(sizeof(FRONT) * maxPoints);
	//fr = malloc(sizeof(FRONT) * nDimensions);
	
	struct timeval tv1, tv2;
	struct rusage ru_before, ru_after;
	getrusage (RUSAGE_SELF, &ru_before);
	
	//process each front to get the hypervolumes
	for (int i = 0; i < f->nFronts; i++) {
		iteration = 0;							//initialise iteration
		printf("Calculating Hypervolume for Front:%d...\n", i+1);
		printf("\t\t\t\t\t%1.10f\n", hypervolume(f->fronts[i]));
		
	}
	
	getrusage (RUSAGE_SELF, &ru_after);
	tv1 = ru_before.ru_utime;
	tv2 = ru_after.ru_utime;
	printf("Average time = %fs\n", (tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6) / f->nFronts);
	
	//free the storage
	for (int i = 0; i < maxDepth+1; i++) {
		for (int j = 0; j < maxPoints; j++) {
			free(fr[i].points[j].objectives);
		}
		free(fr[i].points);
	}
	free(fr);
}