/*
 * The entirety of this work is licensed under the Apache License,
 * Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License.
 *
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <network_test.h>

#define BILL 1000000000L

/* the random ring infrastructure */
int random_ring(CommConfig_t *config, int norand, int n_measurements, int nrands, int niters, 
                CommTest_t req_test, CommTest_t other_test, MPI_Comm comm, MPI_Comm global_comm, 
                CommResults_t *results)
{
     int m, r, n, i, nranks, myrank, real_n_measurements, total_vals;
     double timeout_t1, timeout;
     char *all_hnames, left_neighbor_node[MPI_MAX_PROCESSOR_NAME], right_neighbor_node[MPI_MAX_PROCESSOR_NAME];
     int *rank_list;
     double *myperf_vals, *myperf_vals_hires;

     mpi_error(MPI_Comm_size(comm, &nranks));
     mpi_error(MPI_Comm_rank(comm, &myrank));

     /* get the nodes that each rank is on so that perf can be tracked by node */
     all_hnames = malloc(sizeof(char) * nranks * MPI_MAX_PROCESSOR_NAME);
     if (all_hnames == NULL) {
          die("Failed to allocate all_hnames in random_ring()\n");
     }
     mpi_error(MPI_Allgather(config->mynode->host_name, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_hnames, 
                             MPI_MAX_PROCESSOR_NAME, MPI_CHAR, comm));

     /* allocate space for our rank list */
     total_vals        = n_measurements * nrands;
     rank_list         = malloc(sizeof(int) * nranks);
     myperf_vals       = malloc(sizeof(double) * total_vals);
     myperf_vals_hires = malloc(sizeof(double) * total_vals * niters);
     if (rank_list == NULL || myperf_vals == NULL || myperf_vals_hires == NULL) {
          die("Failed to allocate perf_vals in random_ring()\n");
     }

     /* outer loop over "measurements" */
     timeout_t1 = MPI_Wtime();
     real_n_measurements = 0;
     for (m = 0; m < n_measurements; m++) {

          /* check if we need to timeout this test because it is running too long */
          timeout = MPI_Wtime() - timeout_t1;
          mpi_error(MPI_Allreduce(MPI_IN_PLACE, &timeout, 1, MPI_DOUBLE, MPI_MIN, global_comm));
          if (TEST_TIMEOUT_SECS <= timeout) continue;
          real_n_measurements++;

          /* loop over new random lists */
          for (r = 0; r < nrands; r++) {

               /* generate a list of ranks */
               for (n = 0; n < nranks; n++) {
                    rank_list[n] = n;
               }
               if (norand != 1) shuffle(rank_list, nranks, RSEED+config->mynode_rank, r);

               /* determine our left and right neighbors */
               int myn, left_nighbor, right_neigbor;
               for (n = 0; n < nranks; n++) {
                    if (rank_list[n] == myrank) {
                         myn = n;
                         continue;
                    }
               }
               int left_neighbor = myn - 1;
               if (left_neighbor < 0) left_neighbor += nranks;
               int right_neighbor = myn + 1;
               if (right_neighbor >= nranks) right_neighbor -= nranks;
               left_neighbor  = rank_list[left_neighbor];
               right_neighbor = rank_list[right_neighbor];
               memcpy(left_neighbor_node, &all_hnames[left_neighbor*MPI_MAX_PROCESSOR_NAME], MPI_MAX_PROCESSOR_NAME);
               memcpy(right_neighbor_node, &all_hnames[right_neighbor*MPI_MAX_PROCESSOR_NAME], MPI_MAX_PROCESSOR_NAME);

               /* call the requested test */
               double myperf;
               int poffset = m*nrands + r;
               int poffset_hires = m*nrands*niters + r*niters;
               switch (req_test) {
               case P2P_LATENCY:
                    p2p_latency(config, left_neighbor, right_neighbor, niters, global_comm, 
                                comm, &myperf_vals_hires[poffset_hires], &myperf_vals[poffset]);
                    break;
               case P2P_BANDWIDTH:
                    p2p_bandwidth(config, left_neighbor, right_neighbor, niters, global_comm, 
                                  comm, &myperf_vals_hires[poffset_hires], &myperf_vals[poffset]);
                    break;
               case RMA_LATENCY:
                    rma_latency(config, left_neighbor, right_neighbor, niters, global_comm, 
                                comm, &myperf_vals_hires[poffset_hires], &myperf_vals[poffset]);
                    break;
               case RMA_BANDWIDTH:
                    rma_bandwidth(config, left_neighbor, right_neighbor, niters, global_comm, 
                                  comm, &myperf_vals_hires[poffset_hires], &myperf_vals[poffset]);
                    break;
               case P2P_NEIGHBORS:
                    p2p_neighbors(config, left_neighbor, right_neighbor, niters, global_comm, 
                                  comm, &myperf_vals_hires[poffset_hires], &myperf_vals[poffset]);
                    break;
               }

               /* add a record of this test's performance */
               summarize_pairs_performance(config, global_comm, left_neighbor_node, right_neighbor_node, 
                                           &myperf_vals_hires[poffset_hires], niters, m, r, req_test, other_test);

          } // end of random lists

     } // end of measurements

     total_vals /= n_measurements;
     total_vals *= real_n_measurements;
     int from_min = 0;
     if (req_test == P2P_LATENCY || req_test == RMA_LATENCY) from_min = 1;
     summarize_performance(config, myperf_vals_hires, myperf_vals, (total_vals * niters), total_vals, 
                           from_min, global_comm, results);

     free(rank_list);
     free(myperf_vals);
     free(myperf_vals_hires);
     free(all_hnames);

     return 0;
}

int p2p_latency(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, 
                MPI_Comm comm, double *perfvals, double *perfval)
{
     int i;
     struct timespec t1, t2;
     double bt1, bt2;
     MPI_Request *requests;

     requests = malloc(sizeof(MPI_Request) * 4);
     if (requests == NULL) {
          die("Failed to allocate requests in p2p_latency()\n");
     }

     mpi_error(MPI_Barrier(global_comm));
     for (i = -200; i < niters; i++) {
          if (i == 0) bt1 = MPI_Wtime();
          if (i >= 0) clock_gettime(CLOCK_MONOTONIC, &t1);

          mpi_error(MPI_Irecv(&config->p2p_buffer[0], 1, MPI_DOUBLE, lneighbor, 81, comm, &requests[0]));
          mpi_error(MPI_Irecv(&config->p2p_buffer[1], 1, MPI_DOUBLE, rneighbor, 82, comm, &requests[1]));
          mpi_error(MPI_Isend(&config->p2p_buffer[2], 1, MPI_DOUBLE, lneighbor, 82, comm, &requests[2]));
          mpi_error(MPI_Isend(&config->p2p_buffer[3], 1, MPI_DOUBLE, rneighbor, 81, comm, &requests[3]));
          mpi_error(MPI_Waitall(4, requests, MPI_STATUS_IGNORE));
          if (i >= 0) {
               clock_gettime(CLOCK_MONOTONIC, &t2);
               /* we convert to usec and count for the 2 sends */
               perfvals[i] = 5e-4 * (double)(BILL * (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec));
          }

     }
     bt2 = MPI_Wtime();

     *perfval = 1.0e6 * (bt2 - bt1) / (double)(2 * niters);
     mpi_error(MPI_Barrier(comm));
     free(requests);
     return 0;

}

int rma_latency(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, 
                MPI_Comm comm, double *perfvals, double *perfval)
{
     int i, j;
     struct timespec t1, t2;
     double bt1, bt2;

     mpi_error(MPI_Win_lock(MPI_LOCK_SHARED, lneighbor, 0, config->rma_window));
     if (lneighbor != rneighbor) {
          mpi_error(MPI_Win_lock(MPI_LOCK_SHARED, rneighbor, 0, config->rma_window));
     }

     mpi_error(MPI_Barrier(global_comm));
     for (i = -200; i < niters; i++) {
          if (i == 0) bt1 = MPI_Wtime();
          if (i >= 0) clock_gettime(CLOCK_MONOTONIC, &t1);

          mpi_error(MPI_Get(&config->rma_buffer[0], 1, MPI_DOUBLE, lneighbor, 2L, 1, MPI_DOUBLE, config->rma_window));
          mpi_error(MPI_Get(&config->rma_buffer[1], 1, MPI_DOUBLE, rneighbor, 3L, 1, MPI_DOUBLE, config->rma_window));
          mpi_error(MPI_Win_flush(lneighbor, config->rma_window));
          mpi_error(MPI_Win_flush(rneighbor, config->rma_window));
          if (i >= 0) {
               clock_gettime(CLOCK_MONOTONIC, &t2);
               /* we convert to usec and count for the 2 sends */
               perfvals[i] = 5e-4 * (double)(BILL * (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec));
          }

     }
     bt2 = MPI_Wtime();

     *perfval = 1.0e6 * (bt2 - bt1) / (double)(2 * niters);
     mpi_error(MPI_Barrier(comm));
     mpi_error(MPI_Win_unlock(lneighbor, config->rma_window));
     if (lneighbor != rneighbor) {
          mpi_error(MPI_Win_unlock(rneighbor, config->rma_window));
     }

     return 0;

}

int p2p_bandwidth(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, 
                  MPI_Comm comm, double *perfvals, double *perfval)
{
     int i, j;
     struct timespec t1, t2;
     double bt1, bt2;
     MPI_Request *requests;

     requests = malloc(sizeof(MPI_Request) * 4 * config->bw_outstanding);
     if (requests == NULL) {
          die("Failed to allocate requests in p2p_bandwidth()\n");
     }

     mpi_error(MPI_Barrier(global_comm));
     for (i = -1; i < niters; i++) {
          if (i == 0) bt1 = MPI_Wtime();
          if (i >= 0) clock_gettime(CLOCK_MONOTONIC, &t1);

          for (j = 0; j < config->bw_outstanding; j++) {
               mpi_error(MPI_Irecv(&config->p2p_buffer[2*config->p2pbw_cnt], config->p2pbw_cnt, MPI_DOUBLE, 
                                   lneighbor, 81, comm, &requests[j]));
               mpi_error(MPI_Irecv(&config->p2p_buffer[3*config->p2pbw_cnt], config->p2pbw_cnt, MPI_DOUBLE, 
                                   rneighbor, 82, comm, &requests[j+config->bw_outstanding]));
          }
          for (j = 0; j < config->bw_outstanding; j++) {
               mpi_error(MPI_Isend(&config->p2p_buffer[0], config->p2pbw_cnt, MPI_DOUBLE, lneighbor, 82, 
                                   comm, &requests[j+2*config->bw_outstanding]));
               mpi_error(MPI_Isend(&config->p2p_buffer[config->p2pbw_cnt], config->p2pbw_cnt, MPI_DOUBLE, 
                                   rneighbor, 81, comm, &requests[j+3*config->bw_outstanding]));
          }
          mpi_error(MPI_Waitall(4*config->bw_outstanding, requests, MPI_STATUS_IGNORE));
          if (i >= 0) {
               clock_gettime(CLOCK_MONOTONIC, &t2);
               perfvals[i] = 1e-9 * (double)(BILL * (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec));
          }

     }
     bt2 = MPI_Wtime();

     /* we report uni-directional BW in MB/s/rank */
     for (i = 0; i < niters; i++) {
          perfvals[i] = (double)(sizeof(config->p2p_buffer[0]) * config->p2pbw_cnt * 
                                 2 * config->bw_outstanding) / (perfvals[i] * 1024. * 1024.);
     }
     *perfval = (double)(sizeof(config->p2p_buffer[0]) * config->p2pbw_cnt * 2 * 
                         niters * config->bw_outstanding) / ((bt2 - bt1) * 1024. * 1024.);
     mpi_error(MPI_Barrier(comm));
     free(requests);
     return 0;

}

int rma_bandwidth(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, 
                  MPI_Comm comm, double *perfvals, double *perfval)
{
     int i, j;
     struct timespec t1, t2;
     double bt1, bt2;

     mpi_error(MPI_Win_lock(MPI_LOCK_SHARED, lneighbor, 0, config->rma_window));
     if (lneighbor != rneighbor) {
          mpi_error(MPI_Win_lock(MPI_LOCK_SHARED, rneighbor, 0, config->rma_window));
     }

     mpi_error(MPI_Barrier(global_comm));
     for (i = -1; i < niters; i++) {
          if (i == 0) bt1 = MPI_Wtime();
          if (i >= 0) clock_gettime(CLOCK_MONOTONIC, &t1);

          for (j = 0; j < config->bw_outstanding; j++) {
               mpi_error(MPI_Put(&config->rma_buffer[0], config->rmabw_cnt, MPI_DOUBLE, lneighbor, 2L*config->rmabw_cnt, 
                                 config->rmabw_cnt, MPI_DOUBLE, config->rma_window));
               mpi_error(MPI_Put(&config->rma_buffer[config->rmabw_cnt], config->rmabw_cnt, MPI_DOUBLE, rneighbor, 
                                 3L*config->rmabw_cnt, config->rmabw_cnt, MPI_DOUBLE, config->rma_window));
          }
          mpi_error(MPI_Win_flush(lneighbor, config->rma_window));
          mpi_error(MPI_Win_flush(rneighbor, config->rma_window));

          if (i >= 0) {
               clock_gettime(CLOCK_MONOTONIC, &t2);
               perfvals[i] = 1e-9 * (double)(BILL * (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec));
          }

     }
     bt2 = MPI_Wtime();

     /* we report uni-directional BW in MB/s/rank */
     for (i = 0; i < niters; i++) {
          perfvals[i] = (double)(sizeof(config->rma_buffer[0]) * config->rmabw_cnt * 2 * 
                                 config->bw_outstanding) / (perfvals[i] * 1024. * 1024.);
     }
     *perfval = (double)(sizeof(config->rma_buffer[0]) * config->rmabw_cnt * 2 * 
                         niters * config->bw_outstanding) / ((bt2 - bt1) * 1024. * 1024.);
     mpi_error(MPI_Barrier(comm));

     mpi_error(MPI_Win_unlock(lneighbor, config->rma_window));
     if (lneighbor != rneighbor) {
          mpi_error(MPI_Win_unlock(rneighbor, config->rma_window));
     }

     return 0;

}

int p2p_neighbors(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, 
                  MPI_Comm comm, double *perfvals, double *perfval)
{
     int i, j;
     struct timespec t1, t2;
     double bt1, bt2;
     MPI_Request *requests;

     requests = malloc(sizeof(MPI_Request) * 4 * config->bw_outstanding);
     if (requests == NULL) {
          die("Failed to allocate requests in p2p_neighbors()\n");
     }

     mpi_error(MPI_Barrier(global_comm));
     for (i = -1; i < niters; i++) {
          if (i == 0) bt1 = MPI_Wtime();
          if (i >= 0) clock_gettime(CLOCK_MONOTONIC, &t1);

          for (j = 0; j < config->bw_outstanding; j++) {
               mpi_error(MPI_Irecv(&config->p2p_buffer[2*config->p2pbw_cnt], config->p2pbw_cnt, MPI_DOUBLE, 
                                   lneighbor, 81, comm, &requests[j]));
               mpi_error(MPI_Irecv(&config->p2p_buffer[3*config->p2pbw_cnt], config->p2pbw_cnt, MPI_DOUBLE, 
                                   rneighbor, 82, comm, &requests[j+config->bw_outstanding]));
          }
          for (j = 0; j < config->bw_outstanding; j++) {
               mpi_error(MPI_Isend(&config->p2p_buffer[0], config->p2pbw_cnt, MPI_DOUBLE, lneighbor, 82, comm, 
                                   &requests[j+2*config->bw_outstanding]));
               mpi_error(MPI_Isend(&config->p2p_buffer[config->p2pbw_cnt], config->p2pbw_cnt, MPI_DOUBLE, 
                                   rneighbor, 81, comm, &requests[j+3*config->bw_outstanding]));
          }

          mpi_error(MPI_Waitall(4*config->bw_outstanding, requests, MPI_STATUS_IGNORE));
          mpi_error(MPI_Barrier(comm));

          if (i >= 0) {
               clock_gettime(CLOCK_MONOTONIC, &t2);
               perfvals[i] = 1e-9 * (double)(BILL * (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec));
          }

     }
     bt2 = MPI_Wtime();

     /* we report uni-directional BW in MB/s/rank */
     for (i = 0; i < niters; i++) {
          perfvals[i] = (double)(sizeof(config->p2p_buffer[0]) * config->p2pbw_cnt * 2 * 
                                 config->bw_outstanding) / (perfvals[i] * 1024. * 1024.);
     }
     *perfval = (double)(sizeof(config->p2p_buffer[0]) * config->p2pbw_cnt * 2 * niters * 
                         config->bw_outstanding) / ((bt2 - bt1) * 1024. * 1024.);
     mpi_error(MPI_Barrier(comm));
     free(requests);
     return 0;
}
