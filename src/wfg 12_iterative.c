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

/* new hypervolume variables */
FRONT *frontsArray;      // the stack of fronts
int *indexStack;        // the indices of the contributing points
double *hvStack;     // partial volumes
double *ehvStack;    // exclusive volumes

//NOte: n is needed for slicing and sorting, iteration is needed for saving array of fronts when going deeper into recursion

/* returns a minimum of two numbers */
#define MIN(x, y) ((x < y) ? (x) : (y))

/**
 * prints the front 
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

/* compare function for qsort sorting front in the last objective, i.e. increasing from top to bottom */
/* and we process hypervolumes from the bottom */
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
	for (int i = front.nPoints - 1; i >= 0; i--) // process from bottom (highest last objective)
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
	//allocate an array of fronts for nDimensions (not needed anymore)
	fr = malloc(sizeof(FRONT) * maxd);
	for (int i = 0; i < maxd; i++) {
		fr[i].points = malloc(sizeof(POINT) * maxPoints);
		for (int j = 0; j < maxPoints; j++) {
		//only need to malloc for n (number of Dimensions)
			fr[i].points[j].objectives = malloc(sizeof(OBJECTIVE) * (maxDimensions - (i+1)));
		}
	}
	
	// allocate memory
	frontsArray = malloc(sizeof(FRONT) * (maxDimensions /* - 1 */));
	for (int i = 1; i < maxDimensions /* - 1 */; i++) {
	frontsArray[i].points = malloc(sizeof(POINT) * maxPoints);
		for (int j = 0; j < maxPoints; j++) {
			frontsArray[i].points[j].objectives = malloc(sizeof(OBJECTIVE) * (maxDimensions /* - (i + 1) */)); 
		}
	}
	indexStack  = malloc(sizeof(int)    * (maxDimensions /* - 2 */));
	ehvStack = malloc(sizeof(double) * (maxDimensions /* - 2 */));
	hvStack  = malloc(sizeof(double) * (maxDimensions /* - 1 */));

	//process each front to get the hypervolumes
	for (int i = 0; i < f->nFronts; i++) {
		frontsArray[0] = f->fronts[i];
		n = f->fronts[i].n;
		hvnew(); 
		printf("Calculating Hypervolume for Front:%d...\n", i+1);
		printf("\t\t\t\t\t%1.10f\n", hvStack[0]);
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
