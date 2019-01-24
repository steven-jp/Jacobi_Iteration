#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <assert.h>

#define SIZE 1024

typedef struct {
  int numth;
  int mynum;
  int rowcount;
  double threshcheck;
  double maxthresh;
  int start;
  int end;
  double (*old)[SIZE];
  double (*new)[SIZE];
} thd_t;


typedef struct {
  int totalthreads;
  int threadsarrived;
  int release;
  pthread_mutex_t mtx;
  pthread_cond_t ready;
} barrier;


/* barrier methods */
void barrier_init();
void barrier_arrived();
void barrier_free();

/* matrix methods */
void init_matrix(double (*omtx)[SIZE]);
void* fill_matrix(void* thdargs);
void output_matrix(double (*omtx)[SIZE], FILE* out);

/* globals */
double threshold = 0.00001;
int threads = 0;
double* maxarray = NULL;
pthread_mutex_t mtx;
struct timespec start, end;
double elapsedtime;
barrier* b;

int main() {
  int numtests = 10;
  printf("%s\n", "threshold?");
  scanf("%lf",&threshold);
  printf("%s\n", "How many tests?");
  scanf("%i",&numtests);
  for (int k = 0; k < numtests; k++) {
      threads = k+1;
      printf("Threshold: %18.15f, Threads: %i\n", threshold, threads);
      pthread_t thd[threads];

      /* initialize beginning 2 matrices */
      double(*old)[SIZE] = malloc(SIZE*SIZE*sizeof(double));
      double(*new)[SIZE] = malloc(SIZE*SIZE*sizeof(double));
      init_matrix(old);
      for (int i = 0; i < SIZE; i++) {
          for (int j = 0; j < SIZE; j++) {
            new[i][j] = old[i][j];
          }
      }
      clock_gettime(CLOCK_MONOTONIC, &start);

      /* holds maximum values for threshold*/
      maxarray = malloc(threads * sizeof(double));
      for (int i = 0; i < threads; i++) {
        maxarray[i] = 0;
      }
      int rowcount = SIZE/threads;

      /* thread and barrier initialization*/
       barrier_init();
       pthread_mutex_init(&b->mtx,NULL);
       for (int i = 0; i < threads; i++) {
          thd_t* thdargs = malloc(sizeof(thd_t));
          thdargs->numth = threads;
          thdargs->mynum = i;
          thdargs->rowcount = rowcount;
          thdargs->old = old;
          thdargs->new = new;
          if (pthread_create(&thd[i], NULL,fill_matrix, thdargs)){
            perror("pthread create");
            exit(1);
          }
       }
       for (int i = 0; i < threads; i++) {
          if(pthread_join(thd[i],NULL)){
            perror("pthread join");
            exit(1);
          }
        }

      /* print to file */
      FILE* output;
      if ((output = fopen("output.mtx","w")) == NULL) {
        perror("output file");
        exit(1);
      }

      /* end routines */
      output_matrix(old, output);
      fclose(output);
      clock_gettime(CLOCK_MONOTONIC, &end);
      printf("%ld seconds\n", end.tv_sec-start.tv_sec);
      barrier_free();
      free(new);
      free(old);
      free(maxarray);
  }
}

  /* implements jacobi iteration for laplaces equation
   *  till a threshold provided is met.
   */
  void* fill_matrix(void* thdargs) {
    thd_t* targs = (thd_t*) thdargs;

    /* get value range for each thread */
    targs->start = targs->mynum*targs->rowcount+1;
    targs->end = targs->start + targs->rowcount - 1;
    if (targs->end == SIZE-1) {
      targs->end = SIZE-2;
    }
    else if (targs->end == SIZE) {
      targs->end = SIZE-2;
    }

    /* iterate till threshold is met */
    int keepgoing = 1;
    while (keepgoing == 1) {

      /* copy to another matrix*/
      for (int i = targs->start; i <= targs->end; i++) {
        for (int j = 1; j < SIZE-1; j++) {
          targs->new[i][j] = (targs->old[i-1][j] + targs->old[i+1][j] + targs->old[i][j-1] + targs->old[i][j+1])/4.0;
        }
      }
      barrier_arrived();

      /* copy back to matrix */
      for (int i = targs->start; i <= targs->end; i++) {
        for (int j = 1; j < SIZE-1; j++) {
          targs->old[i][j] = (targs->new[i-1][j] + targs->new[i+1][j] + targs->new[i][j-1] + targs->new[i][j+1])/4.0;
        }
      }
      barrier_arrived();

      /* check the difference between every element and grab the largest */
      targs->maxthresh = -INFINITY;
      for (int i = targs->start; i <= targs->end; i++) {
        for (int j = 1; j < SIZE-1; j++) {
         targs->threshcheck = targs->old[i][j]-targs->new[i][j];
          if (targs->threshcheck > targs->maxthresh) {
            targs->maxthresh = targs->threshcheck;
         }
        }
      }
      maxarray[targs->mynum] = targs->maxthresh;
      barrier_arrived();

      /* check if meets threshold */
      pthread_mutex_lock(&mtx);
      double largest = 0.0;
      double smallest = 0.0;
      for (int i = 0; i < threads; i++) {
        if (maxarray[i] > largest) {
          largest = maxarray[i];
        }
        if (maxarray[i] < smallest) {
          smallest = maxarray[i];
        }
      }
      if (largest - smallest == 0){
        ;
      }
      else if  (largest-smallest <= threshold) {
        keepgoing = 0;
      }
      pthread_mutex_unlock(&mtx);
    }
    return (void*) 1;
}

/* output matrix values to file given
 */
void output_matrix(double (*new)[SIZE], FILE* output){
  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      fprintf(output,"%11.10f ", new[i][j]);
    }
  }
}

/* initialized a matrix with file input
 */
void init_matrix(double (*old)[SIZE]){
  FILE* file = fopen("input.mtx", "r");
  for (int i = 0; i < SIZE; i++){
    for (int j = 0; j < SIZE; j++){
      fscanf(file,"%lf", &old[i][j]);
    }
  }
  fclose(file);
}


/* initialization of a barrier
 *
 */
void barrier_init(){
  b = malloc(sizeof(barrier));
  b->totalthreads = threads;
  b->threadsarrived = 0;
  pthread_mutex_init(&b->mtx,NULL);
  pthread_cond_init(&b->ready,NULL);
}

/* implementation of a barrier
 *
 */
void barrier_arrived(){
    pthread_mutex_lock(&(b->mtx));
    b->threadsarrived++;
    if(b->threadsarrived == b->totalthreads){
        b->threadsarrived = 0;
        pthread_cond_broadcast(&(b->ready));
    }
    else {
      pthread_cond_wait(&(b->ready),&(b->mtx));
    }
    pthread_mutex_unlock(&b->mtx);
}

/* free barrier created
 *
 */
void barrier_free(){
  assert(b != NULL);
  pthread_cond_destroy(&(b->ready));
  pthread_mutex_destroy(&(b->mtx));
  free(b);
}
