
//an objective is a value needed to be optimised
typedef double OBJECTIVE;

//each point has a set number of objectives.
typedef struct
{
	OBJECTIVE *objectives;
} POINT;

//a front structure contains number of points, pointer to the points, and n is the number of objectives in the front (4) 
typedef struct
{
	int nPoints;
	int n;
	POINT *points;
} FRONT;

typedef struct
{
        FRONT sprime;   // reduced front 
        int id;         // index in the original list 
        int k;          // next segment to be evaluated 
        double partial; // volume so far 
        int left;       // left child in the heap 
        int right;      // right child in the heap 
} JOB;

