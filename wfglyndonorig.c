// To do: 
// - can we sort less often or reduce/optimise dominance checks? 

#include "read.c"
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "wfg.h"

int n;     // the number of objectives 
POINT ref; // the reference point 

#define BEATS(x,y)   (x >  y) // change this for max/minimisation

#define WORSE(x,y)   (BEATS(y,x) ? (x) : (y)) 

FRONT *fs;      // memory management stuff 
int fr = 0;     // current depth 

double hv(FRONT);


int greater(const void *v1, const void *v2)
// this sorts points improving in the last objective
{
  POINT p = *(POINT*)v1;
  POINT q = *(POINT*)v2;
  for (int i = n - 1; i >= 0; i--)
    if BEATS(p.objectives[i],q.objectives[i]) return  1;
    else
    // can we manage with only one comparison?
    if BEATS(q.objectives[i],p.objectives[i]) return -1;
  return 0;
}


int dominates2way(POINT p, POINT q)
// returns -1 if p dominates q, 1 if q dominates p, 2 if p == q, 0 o/w
{
  int z = 2;
  // domination could be checked in either order 
  for (int i = n - 1; i >= 0; i--)
    if BEATS(p.objectives[i],q.objectives[i]) 
      {if (z ==  1) return 0; else z = -1;}
    else
    // can we manage with only one comparison?
    if BEATS(q.objectives[i],p.objectives[i]) 
      {if (z == -1) return 0; else z =  1;}
  return z;
}


void limitFront(FRONT ps, int p)
// creates the front ps[p+1 ..] in fs[fr], 
// with each point bounded by ps[p] 
{
  for (int i = 0; i < ps.nPoints - 1 - p; i++)
    for (int j = 0; j < n; j++) 
      fs[fr].points[i].objectives[j] = WORSE(ps.points[p].objectives[j],
                                             ps.points[p+1+i].objectives[j]); 
}


void loseDominated(int z)
// loses dominated points from fs[fr][0 .. z-1] 
{
  POINT t; // have to do proper swaps because of the reuse of the memory hierarchy 
  int j; bool keep;
  fs[fr].nPoints = 1;
  for (int i = 1; i < z; i++)
    {j = 0; keep = true;
     while (j < fs[fr].nPoints && keep)
       switch (dominates2way(fs[fr].points[i], fs[fr].points[j]))
	 {case -1: fs[fr].nPoints--; 
                   t = fs[fr].points[j];
                   fs[fr].points[j] = fs[fr].points[fs[fr].nPoints]; 
                   fs[fr].points[fs[fr].nPoints] = t; 
                   break;
          case  0: j++; break;
          // case  2: printf("Identical points!\n");
	  default: keep = false;
	 }
     if (keep) {t = fs[fr].points[fs[fr].nPoints]; 
                fs[fr].points[fs[fr].nPoints] = fs[fr].points[i]; 
                fs[fr].points[i] = t; 
                fs[fr].nPoints++;}
    }
}


double hv2(FRONT ps)
// returns the hypervolume of ps[0 ..] in 2D 
// assumes that ps is sorted improving
{
  double volume = fabs((ps.points[0].objectives[0] - ref.objectives[0]) * 
                       (ps.points[0].objectives[1] - ref.objectives[1])); 
  for (int i = 1; i < ps.nPoints; i++) 
    volume += fabs((ps.points[i].objectives[0] - ref.objectives[0]) * 
                   (ps.points[i].objectives[1] - ps.points[i - 1].objectives[1]));
  return volume;
}


double inclhv(POINT p)
// returns the inclusive hypervolume of p
{
  double volume = 1;
  for (int i = 0; i < n; i++) 
    volume *= fabs(p.objectives[i] - ref.objectives[i]);
  return volume;
}


double exclhv(FRONT ps, int p)
// returns the exclusive hypervolume of ps[p] relative to ps[p+1 ..] 
{
  double volume = inclhv(ps.points[p]);
  if (ps.nPoints > p + 1) 
    {limitFront(ps, p);
     loseDominated(ps.nPoints - 1 - p);
     fr++;
     volume -= hv(fs[fr - 1]);
     fr--;}
  return volume;
}


double hv(FRONT ps)
// returns the hypervolume of ps[0 ..] 
{
  qsort(ps.points, ps.nPoints, sizeof(POINT), greater);
  if (n == 2) return hv2(ps);

  double volume = 0;
  n--;
  for (int i = ps.nPoints - 1; i >= 0; i--)
    // we can ditch dominated points here, 
    // but they will be ditched anyway in dominatedBit 
    volume += fabs(ps.points[i].objectives[n] - ref.objectives[n]) * exclhv(ps, i);
  n++; 

  return volume;
}


int main(int argc, char *argv[]) 
// processes each front from the file 
{
  FILECONTENTS *f = readFile(argv[1]);

  struct timeval tv1, tv2;
  struct rusage ru_before, ru_after;
  getrusage (RUSAGE_SELF, &ru_before);

  // find the biggest fronts
  int maxm = 0;
  int maxn = 0;
  for (int i = 0; i < f->nFronts; i++)
    {if (f->fronts[i].nPoints > maxm) maxm = f->fronts[i].nPoints;
     if (f->fronts[i].n       > maxn) maxn = f->fronts[i].n;}

  // allocate memory
  int maxd = maxn - 2; 
  fs = malloc(sizeof(FRONT) * maxd);
  for (int i = 0; i < maxd; i++) 
    {fs[i].points = malloc(sizeof(POINT) * maxm); 
     for (int j = 0; j < maxm; j++) 
       fs[i].points[j].objectives = malloc(sizeof(OBJECTIVE) * (maxn - (i + 1)));}

  // initialise the reference point
  ref.objectives = malloc(sizeof(OBJECTIVE) * maxn);
  if (argc == 2)
    {printf("No reference point provided: using the origin\n");
     for (int i = 0; i < maxn; i++) ref.objectives[i] = 0;}
  else if (argc - 2 != maxn)
    {printf("Your reference point should have %d values\n", maxn);
     return 0;}
  else 
  for (int i = 2; i < argc; i++) ref.objectives[i - 2] = atof(argv[i]);

  for (int i = 0; i < f->nFronts; i++) 
    {n = f->fronts[i].n;
     printf("hv(%d) = %1.10f\n", i+1, hv(f->fronts[i]));}

  getrusage (RUSAGE_SELF, &ru_after);
  tv1 = ru_before.ru_utime;
  tv2 = ru_after.ru_utime;

  printf("Average time = %fs\n", (tv2.tv_sec + tv2.tv_usec * 1e-6 - tv1.tv_sec - tv1.tv_usec * 1e-6) / f->nFronts);
  return 0;
}
