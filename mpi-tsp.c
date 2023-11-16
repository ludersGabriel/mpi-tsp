/* WSCAD - 9th Marathon of Parallel Programming
 * Simple Brute Force Algorithm for the
 * Traveling-Salesman Problem
 * Author: Emilio Francesquini - francesquini@ic.unicamp.br
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include <sys/time.h>
#include <time.h>

int min_distance;
int nb_towns;
int world_rank;
int world_size;

typedef struct {
  int to_town;
  int dist;
} d_info;

d_info **d_matrix;
int *dist_to_origin;

int present(int town, int depth, int *path) {
  int i;
  for (i = 0; i < depth; i++)
    if (path[i] == town) return 1;
  return 0;
}

void tsp(int depth, int current_length, int *path) {
  int i;
  if (current_length >= min_distance) return;
  if (depth == nb_towns) {
    current_length += dist_to_origin[path[nb_towns - 1]];
    if (current_length < min_distance) min_distance = current_length;
  } else {
    int town, me, dist;
    me = path[depth - 1];
    for (i = 0; i < nb_towns; i++) {
      town = d_matrix[me][i].to_town;
      if (!present(town, depth, path)) {
        path[depth] = town;
        dist = d_matrix[me][i].dist;
        tsp(depth + 1, current_length + dist, path);
      }
    }
  }
}

void greedy_shortest_first_heuristic(int *x, int *y) {
  int i, j, k, dist;
  int *tempdist;

  tempdist = (int *)malloc(sizeof(int) * nb_towns);
  // Could be faster, albeit not as didactic.
  // Anyway, for tractable sizes of the problem it
  // runs almost instantaneously.
  for (i = 0; i < nb_towns; i++) {
    for (j = 0; j < nb_towns; j++) {
      int dx = x[i] - x[j];
      int dy = y[i] - y[j];
      tempdist[j] = dx * dx + dy * dy;
    }
    for (j = 0; j < nb_towns; j++) {
      int tmp = INT_MAX;
      int town = 0;
      for (k = 0; k < nb_towns; k++) {
        if (tempdist[k] < tmp) {
          tmp = tempdist[k];
          town = k;
        }
      }
      tempdist[town] = INT_MAX;
      d_matrix[i][j].to_town = town;
      dist = (int)sqrt(tmp);
      d_matrix[i][j].dist = dist;
      if (i == 0) dist_to_origin[town] = dist;
    }
  }

  free(tempdist);
}

void init_tsp() {
  int i, st;
  int *x, *y;

  min_distance = INT_MAX;

  if (world_rank == 0) st = scanf("%u", &nb_towns);

  MPI_Bcast(&nb_towns, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&st, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (st != 1) exit(1);

  d_matrix = (d_info **)malloc(sizeof(d_info *) * nb_towns);
  for (i = 0; i < nb_towns; i++)
    d_matrix[i] = (d_info *)malloc(sizeof(d_info) * nb_towns);
  dist_to_origin = (int *)malloc(sizeof(int) * nb_towns);

  x = (int *)malloc(sizeof(int) * nb_towns);
  y = (int *)malloc(sizeof(int) * nb_towns);

  int quit = 0;

  if (world_rank == 0) {
    for (i = 0; i < nb_towns; i++) {
      st = scanf("%u %u", x + i, y + i);

      if (st != 2) {
        quit = 1;
        break;
      };
    }
  }

  MPI_Bcast(&quit, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(x, nb_towns, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(y, nb_towns, MPI_INT, 0, MPI_COMM_WORLD);

  if (quit) exit(1);

  greedy_shortest_first_heuristic(x, y);

  free(x);
  free(y);
}

void printAllDataToLogFile() {
  char filename[100];
  sprintf(filename, "log_%d.txt", world_rank);
  FILE *out = fopen(filename, "w");

  fprintf(out, "world_rank: %d\n", world_rank);
  fprintf(out, "world_size: %d\n", world_size);
  fprintf(out, "nb_towns: %d\n", nb_towns);

  for (int i = 0; i < nb_towns; i++) {
    for (int j = 0; j < nb_towns; j++) {
      fprintf(out, "%d ", d_matrix[i][j].to_town);
      fprintf(out, "%d ", d_matrix[i][j].dist);
    }
    fprintf(out, "\n");
  }

  fclose(out);
}

int run_tsp() {
  int i, *path;

  init_tsp();

  min_distance = INT_MAX;
  path = (int *)malloc(sizeof(int) * nb_towns);

  path[0] = 0;
  for (int i = world_rank + 1; i < nb_towns; i += world_size) {
    path[1] = i;

    tsp(2, dist_to_origin[i], path);
  }

  int global_min_distance;
  MPI_Allreduce(&min_distance, &global_min_distance, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);

  if (world_rank == 0) printf("%d\n", global_min_distance);

  free(path);
  for (i = 0; i < nb_towns; i++) free(d_matrix[i]);
  free(d_matrix);

  return min_distance;
}

int main(int argc, char **argv) {
  struct timeval start, end;
  double totalTime;

  gettimeofday(&start, NULL);

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int num_instances, st;

  if (world_rank == 0) st = scanf("%u", &num_instances);

  MPI_Bcast(&num_instances, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&st, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (st != 1) exit(1);

  while (num_instances-- > 0) run_tsp();

  MPI_Finalize();

  gettimeofday(&end, NULL);

  if (world_rank == 0) {
    totalTime =
        (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    printf("TotalTime: %lf\n", totalTime);
  }

  return 0;
}
