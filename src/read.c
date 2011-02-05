#include "wfg.h"

//trims the nullspace and carriage return character
static void trimLine(char line[])
{
	int i = 0;

	while(line[i] != '\0')
	{
		if (line[i] == '\r' || line[i] == '\n')
		{
			line[i] = '\0';
			break;
		}
		i++;
	}
}

//print the contents of the struct filecontents
void printContents(FILECONTENTS *f)
{
	for (int i = 0; i < f->nFronts; i++)
	{
		printf("Front %d:\n", i+1);
		for (int j = 0; j < f->fronts[i].nPoints; j++)
		{
			printf("\t");
			for (int k = 0; k < f->fronts[i].n; k++)
			{
				printf("%f ", f->fronts[i].points[j].objectives[k]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

//reads the file and insert the information into a filecontents structure
FILECONTENTS *readFile(char filename[])
{
	FILE *fp;
	char line[BUFSIZ];
	
	//signify which front, point, and objective you're working on
	int front = 0, point = 0, objective = 0;

	//allocate a filecontents structure
	FILECONTENTS *fc = (FILECONTENTS *) malloc(sizeof(FILECONTENTS));
	fc->nFronts = 0;
	fc->fronts = NULL;

	fp = fopen(filename, "r");
	if (fp == NULL)
	{
		fprintf(stderr, "File %s could not be opened\n", filename);
		exit(EXIT_FAILURE);
	}

	while(fgets(line, sizeof line, fp) != NULL)
	{
		trimLine(line);
		//if the line is a hash
		if (strcmp(line, "#") == 0)
		{
			front = fc->nFronts;	//next step of the front you're working on
			fc->nFronts++; //each step is a front
			fc->fronts = (FRONT *) realloc(fc->fronts, sizeof(FRONT) * fc->nFronts);  //reallocate a memory for the whole fronts (20 fronts)
			fc->fronts[front].nPoints = 0;	//initialise the front's number of points to zero
			fc->fronts[front].points = NULL;	//initially no point in the front
		}
		else
		{
			FRONT *f = &fc->fronts[front];		//copies the front pointer
			point = f->nPoints;			//next step of the point you're working on
			f->nPoints++;
			f->points = (POINT *) realloc(f->points, sizeof(POINT) * f->nPoints);
			f->n = 0;
			f->points[point].objectives = NULL;
			char *tok = strtok(line, " \t\n");
			do
			{
				POINT *p = &f->points[point];  //copies the point pointer
				objective = f->n;			//next step of the objective you're working on	
				f->n++;
				p->objectives = (OBJECTIVE *) realloc(p->objectives, sizeof(OBJECTIVE) * f->n);
				p->objectives[objective] = atof(tok);
			} while ((tok = strtok(NULL, " \t\n")) != NULL); //while there is still an objective keep reading
		}
	}

	//remove the last hash count
	fc->nFronts--;
	// for (int i = 0; i < fc->nFronts; i++) fc->fronts[i].n = fc->fronts[i].points[0].nObjectives;
        fclose(fp);
	/* printf("Read %d fronts\n", fc->nFronts);
	   printContents(fc); */
	return fc;
}
