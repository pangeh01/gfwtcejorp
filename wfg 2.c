#include "read.c"

int nDimensions;
int iteration;
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
int dominated(POINT a, POINT b, int dimension) {
	//point a is dominated if point b is at least equal in all objectives and better in at least one
	int result = 2;
	for (int i = 0; i < dimension; i++) {
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
	int eliminated[front->nPoints];

	for (int k = 0; k < front->nPoints; k++) {
		eliminated[k] = -1;
		//printf("%d\n", eliminated[k]);
	}
	
	for (int i = 0; i < front->nPoints; i++) {
		for (int j = 0; j < front->nPoints; j++) {
			if(eliminated[j] == 1) 
				continue;
			
			//if the point is dominated by any other point continue
			if (dominated(front->points[i], front->points[j], front->n) == (1 || 0)) {
				eliminated[i] = 1;
				break;
			} else {
				if (j == front->nPoints-1) {
					eliminated[i] = 0;
				}
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
}

//returns an sprimed front relative to point p
FRONT *limitset(int index, FRONT front) {

	FRONT *sprimed = malloc(sizeof(FRONT));
	sprimed->n = front.n;
	sprimed->nPoints = 0;
	sprimed->points = malloc(sizeof(POINT)*front.nPoints);
	
	for (int i = index+1; i < front.nPoints; i++) {
		sprimed->points[i-index-1].objectives = malloc(sizeof(OBJECTIVE) * front.n);
		sprimed->nPoints++;
		for (int j = 0; j < front.n; j++) {
			sprimed->points[i-index-1].objectives[j] = min(front.points[index].objectives[j], front.points[i].objectives[j]);
		}
	}
	
	nondom(sprimed);
	iteration++;
	return sprimed;
}

//returns a sprimed front relative to point p
/*void limitset(int index, FRONT front) {
	
	int sprimedPoints = front.nPoints-1-index;
	for (int i = 0; i < sprimedPoints; i++) {
		for (int j = 0; j < front.n; j++) {
			fr[iteration].points[i].objectives[j] = min(front.points[index].objectives[j], front.points[i+1+index].objectives[j]);
		}
	}
	fr[iteration].nPoints = sprimedPoints;
	
	printfront(fr[iteration]);
	nondom(&fr[iteration]);
	
	iteration++;
	//return &fr[iteration-1];
}*/

/*void sortfront(FRONT *front, int iteration) {
	
	for (int i = 0; i < front->nPoints; i++) {
		for (int j = i+1; j < front->nPoints; j++) {
			if (front->points[i].objectives[front->n-iteration] > front->points[j].objectives[front->n-iteration]) {
				//swap the points
				POINT temp = front->points[i];
				front->points[i] = front->points[j];
				front->points[j] = temp;
			}
		}
	}
}*/

int compare (const void *a, const void *b)
{
	if (((*(POINT *)a).objectives[nDimensions-iteration-1] - (*(POINT *)b).objectives[nDimensions-iteration-1]) > 0)
		return 1;
	else
		return -1;
}

//returns a size of exclusive hypervolume of point p relative to a front set
double ehv(int index, FRONT front) {
	
	double ehv = 1;
	for (int i = 0; i < front.n; i++) {
		ehv *= front.points[index].objectives[i];
	}
	
	//calculate the exclusive hypervolume
	if (front.nPoints > index+1) {
		FRONT *nondomsprimed = limitset(index, front);
		//limitset(index, front);
		ehv -= hypervolume(*nondomsprimed);
		iteration--;
	}

	return ehv;
}

//returns a size hypervolume for a set of points (a front)
double hypervolume(FRONT front) {
	
	double sumhv = 0;
	
	//iteration++;
	
	qsort(front.points, front.nPoints, sizeof (POINT), compare);
	
	//sum all the segments
	for (int i = 0; i < front.nPoints; i++) {
		sumhv += ehv(i, front);
	}
	
	//iteration--;
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
	
	for (int i = 0; i < nDimensions; i++) {
		freefront(&fr[i]);
	}
	free(fr);
	
	getrusage (RUSAGE_SELF, &ru_after);
	tv1 = ru_before.ru_utime;
	tv2 = ru_after.ru_utime;
	printf("Average time = %fs\n", (tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6) / f->nFronts);
}