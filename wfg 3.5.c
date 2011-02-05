#include "read.c"
#include <stdbool.h>

int nDimensions;
int iteration = 0;
extern double ehv(int index, FRONT front);
extern double hypervolume(FRONT front);
FRONT *fr;

//returns a minimum of two numbers
double min(double a, double b) {
	if (a <= b) 
		return a;
	else
		return b;
}

//prints the front 
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


//frees the memory allocated for a front
void freefront(FRONT *front) {
	for (int i = 0; i < front->nPoints; i++) {
		free(front->points[i].objectives);
	}
	free(front->points);
	free(front);
}

//returns 1 if point a, is dominated by point b
//zero if equals
//returns -1 if point b is dominated by a
int dominated(POINT a, POINT b) {
	//point a is dominated if point b is at least equal in all objectives and better in at least one
	int result = 2;
	for (int i = 0; i < nDimensions; i++) {
		if (a.objectives[i] > b.objectives[i]) {
			if (result != 1) result = -1; else return 0;
		}
		if (b.objectives[i] > a.objectives[i]) {
			if (result != -1) result = 1; else return 0;
		}
	}
	if (result == 2) return 0; else return result;
	
}

//returns a non-dominating result of a sprimed front
void nondom(FRONT *front) {
	//an array indicating which points has been eliminated
	//an array indicating which points has been eliminated
	int eliminated[front->nPoints];
	
	for (int k = 0; k < front->nPoints; k++) {
		eliminated[k] = -1;
		//printf("%d\n", eliminated[k]);
	}
	
	for (int i = 0; i < front->nPoints; i++) {
		int new = 0;
		for (int j = 0; j < front->nPoints; j++) {
			if(eliminated[j] == 1) 
				continue;
		
			if (dominated(front->points[i], front->points[j]) == (0)) {
				continue;
			} 
			
			//if the point is dominated by any other point continue
			if (dominated(front->points[i], front->points[j]) == (1)) {
				eliminated[i] = 1;
				break;
			} 
			
			if (dominated(front->points[i], front->points[j]) == 2) {
				new++;
				if (new == 1) {
					eliminated[i] = 0;
					break;
				}
				else {
					eliminated[i] = 1;
					break;
				}
			}
			
			if (j == front->nPoints-1) {
				eliminated[i] = 0;
			}
			
		}
	}
	
	int count = 0;
	for (int i = 0; i < front->nPoints; i++) {
		if (eliminated[i] == 0) {
			front->points[count] = front->points[i];
			count++;
		}
	}
	
	front->nPoints = count;
	
	//make the sprimed front into a non-dominating front
	/*POINT temp;					//temporary buffer
	int n = front->nPoints;
	front->nPoints = 1;	//non-dominating storage set initially have at least 1 point (the first point is stored)
	
	//loop through each point in the sprimed set, starting from the second point
	//[i] refer to the points in sprimed set, [j] refer to the storaged (non-dominating) set 
	for (int i = 1; i < n; i++) {
		int j = 0;
		bool store = true;		//whether to store the point in memory
		
		//checking against every point in storaged set
		while (j < front->nPoints && store) {
			switch (dominated(front->points[i], front->points[j])) {
				case -1:
					//if point a dominates b, discard point b
					temp = front->points[j];
					front->points[j] = front->points[front->nPoints - 1];
					front->points[fr[iteration].nPoints - 1] = temp;
					front->nPoints--;
					break;
				case  0:
					//if a and b non-dominating, continue checking
					j++;
					break;
				default:
					//if b dominates a (1), or a == b (2), point a is not in non-dominating set (stop checking)
					store = false;
			}
		}
		if (store) {
			//add the point a to the storaged set
			temp = front->points[front->nPoints];
			front->points[front->nPoints] = front->points[i]; 
			front->points[i] = temp;
			front->nPoints++;
		}
    }*/
}

//returns a sprimed front relative to point p
void limitset(int index, FRONT front) {

	fr[iteration].nPoints = 0;
	for (int i = 0; i < front.nPoints-1-index; i++) {
		fr[iteration].nPoints++;
		for (int j = 0; j < front.n; j++) {
			fr[iteration].points[i].objectives[j] = min(front.points[index].objectives[j], front.points[i+1+index].objectives[j]);
		}
	}
	
	nondom(&fr[iteration]);
}

int compare (const void *a, const void *b)
{
	if (((*(POINT *)a).objectives[nDimensions-iteration] - (*(POINT *)b).objectives[nDimensions-iteration]) > 0)
		return 1;
	else
		return -1;
}

/*double ehv(int index, FRONT front) {
	
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
}*/

//returns a size of exclusive hypervolume of point p relative to a front set
double ehv(int index, FRONT front) {
	
	FRONT frontp;
	frontp.nPoints = 1;
	frontp.n = front.n;
	frontp.points = &front.points[index];
	
	/*double pointhyp = 1;
	 for (int i = 0; i < front.n; i++) {
	 pointhyp = pointhyp*front.points[index].objectives[i];
	 }*/
	
	/*if (front.nPoints == 1) 
	 return pointhyp;*/
	
	double ehv = hypervolume(frontp);
	
	//calculate the exclusive hypervolume
	if (front.nPoints > index+1) {
		limitset(index, front);
		iteration++;
		ehv = hypervolume(frontp) - hypervolume(fr[iteration-1]);
		iteration--;
	}

	return ehv;
}

//returns a size hypervolume for a set of points (a front)
double hypervolume(FRONT front) {

	double sumhv = 0;
	
	if (front.nPoints == 1) {
		sumhv = 1;
		for (int i = 0; i < front.n; i++) {
			sumhv = sumhv*front.points[0].objectives[i];
		}
		//printf("%f\n", sumhv);
		return sumhv;
	}
	qsort(front.points, front.nPoints, sizeof (POINT), compare);
	
	//sum all the segments
	for (int i = 0; i < front.nPoints; i++) {
		sumhv = sumhv + ehv(i, front);
	}
	return sumhv;
}

int main(int argc, char *argv[]) {
	struct timeval tv1, tv2;
	struct rusage ru_before, ru_after;
	getrusage (RUSAGE_SELF, &ru_before);
	
	FILECONTENTS *f = readFile(argv[1]);
	
	nDimensions = f->fronts[0].n;
	
	//allocate an array of fronts for nDimensions. Dataset can have different points so you have to check the front that has the maximum no of points
	fr = malloc(sizeof(FRONT) * nDimensions);
	for (int i = 0; i < nDimensions; i++) {
		fr[i].nPoints = f->fronts[0].nPoints;
		fr[i].n = nDimensions;
		fr[i].points = malloc(sizeof(POINT) * fr[i].nPoints);
		for (int j = 0; j < fr[i].nPoints; j++) {
			fr[i].points[j].objectives = malloc(sizeof(OBJECTIVE) * fr[i].n);
		}
	}
	
	for (int i = 0; i < f->nFronts; i++) {
		iteration = 0;							//initialise iteration
		printf("Calculating Hypervolume for Front:%d...\n", i+1);
		printf("\t\t\t\t\t%f\n", hypervolume(f->fronts[i]));
		
	}
	
	/*for (int i = 0; i < nDimensions; i++) {
		freefront(&fr[i]);
	}
	free(fr);*/
	
	getrusage (RUSAGE_SELF, &ru_after);
	tv1 = ru_before.ru_utime;
	tv2 = ru_after.ru_utime;
	printf("Average time = %fs\n", (tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6) / f->nFronts);
}