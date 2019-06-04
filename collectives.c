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

/* define where baseline sizes for congestor tests are */
#define A2A_BASE_NODES 64

int a2a_test(CommConfig_t *config, int ntests, int base_niters, MPI_Comm a2acomm, 
             MPI_Comm global_comm, CommResults_t * results)
{
     double *sbuf, *rbuf;
     int i, j, nranks, niters, ntot_iters;
     double bt1, bt2, bt, perfval[1], *perfvals;
     struct timespec t1, t2;
     double timeout_t1, timeout;

     a2a_buffers(config, a2acomm);
     mpi_error(MPI_Comm_size(a2acomm, &nranks));

     /* we assume the requested iterations at a fixed node count and decrease from there linearly to save time */
     niters = base_niters;
     if (nranks > A2A_BASE_NODES) {
          niters = (int)((float)niters *  (float)A2A_BASE_NODES / (float)nranks);
          niters = (niters < 1) ? 1 : niters;
     }
     perfvals = malloc(sizeof(double) * ntests * niters);
     if (perfvals == NULL) {
          die("Failed to allocate perfvals in a2a_test()\n");
     }

     timeout_t1 = MPI_Wtime();
     timeout    = 0.;
     ntot_iters = 0;
     bt = 0.;
     for (j = 0; j < ntests; j++) {

          /* check if we need to timeout this test because it is running too long */
          timeout = MPI_Wtime() - timeout_t1;
          mpi_error(MPI_Allreduce(MPI_IN_PLACE, &timeout, 1, MPI_DOUBLE, MPI_MIN, global_comm));
          if (TEST_TIMEOUT_SECS <= timeout) continue;

          for (i = -1; i < niters; i++) {
               if (i == 0) bt1 = MPI_Wtime();
               if (i >= 0) clock_gettime(CLOCK_MONOTONIC, &t1);

               mpi_error(MPI_Alltoall(config->a2a_sbuffer, config->a2a_cnt, MPI_DOUBLE, config->a2a_rbuffer, 
                                      config->a2a_cnt, MPI_DOUBLE, a2acomm));
               if (i >= 0) {
                    clock_gettime(CLOCK_MONOTONIC, &t2);
                    perfvals[j*niters + i] = 1e-9 * (double)(BILL * (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec));
               }

          }
          bt2 = MPI_Wtime();
          bt += bt2 - bt1;
          ntot_iters += niters;
     }

     /* we report uni-directional BW in MiB/s/rank */
     for (i = 0; i < ntot_iters; i++) {
          perfvals[i] = (double)(sizeof(double) * config->a2a_cnt * (nranks-1)) / (perfvals[i] * 1024. * 1024.);
     }
     perfval[0] = (double)(sizeof(double) * config->a2a_cnt * ntot_iters * (nranks-1)) / (bt * 1024. * 1024.);
     mpi_error(MPI_Barrier(global_comm));

     summarize_performance(config, perfvals, perfval, ntot_iters, 1, 0, global_comm, results);

     free(perfvals);

     return 0;
}

int allreduce_test(CommConfig_t *config, int ntests, int niters, MPI_Comm comm, 
                   MPI_Comm global_comm, CommResults_t * results)
{
     double *sbuf, *rbuf;
     int i, j, ntot_iters;
     double bt1, bt2, bt, perfval[1], *perfvals;
     struct timespec t1, t2;
     double timeout_t1, timeout;

     perfvals = malloc(sizeof(double) * ntests * niters);
     if (perfvals == NULL) {
          die("Failed to allocate perfvals in allreduce_test()\n");
     }

     timeout_t1 = MPI_Wtime();
     timeout    = 0.;
     ntot_iters = 0;
     bt = 0.;
     for (j = 0; j < ntests; j++) {

          /* check if we need to timeout this test because it is running too long */
          timeout = MPI_Wtime() - timeout_t1;
          mpi_error(MPI_Allreduce(MPI_IN_PLACE, &timeout, 1, MPI_DOUBLE, MPI_MIN, global_comm));
          if (TEST_TIMEOUT_SECS <= timeout) continue;

          for (i = -1; i < niters; i++) {
               if (i == 0) bt1 = MPI_Wtime();
               if (i >= 0) clock_gettime(CLOCK_MONOTONIC, &t1);

               mpi_error(MPI_Allreduce(&config->p2p_buffer[0], &config->p2p_buffer[1], 1, MPI_DOUBLE, MPI_SUM, comm));
               if (i >= 0) {
                    clock_gettime(CLOCK_MONOTONIC, &t2);
                    perfvals[j*niters + i] = 1e-3 * (double)(BILL * (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec));
               }

          }
          bt2 = MPI_Wtime();
          bt += bt2 - bt1;
          ntot_iters += niters;
     }

     /* we report operation latency */
     perfval[0] = 1.0e6 * bt / (double)ntot_iters;
     mpi_error(MPI_Barrier(global_comm));

     summarize_performance(config, perfvals, perfval, ntot_iters, 1, 1, global_comm, results);

     free(perfvals);

     return 0;
}
