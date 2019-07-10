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
#include <math.h>

#define BILL 1000000000L

int p2p_incast_congestor(CommConfig_t *config, MPI_Comm comm, int myrank, int comm_ranks)
{
     int i;
     MPI_Request *request_list = NULL; 
     request_list = malloc(sizeof(MPI_Request) * comm_ranks);
     if (request_list == NULL) {
          die("Failed to allocate request_list in p2p_incast_congestor()\n");
     }

     if(myrank == 0) {    
       
          for(i=1; i < comm_ranks; i++) {
               mpi_error(MPI_Irecv(&config->a2a_rbuffer[i * config->incast_cnt], config->incast_cnt, 
                                   MPI_DOUBLE, i, 987, comm, &request_list[i-1])); 
          } 
          mpi_error(MPI_Waitall(comm_ranks-1, request_list, MPI_STATUS_IGNORE));

     } else {

          mpi_error(MPI_Send(config->a2a_sbuffer, config->incast_cnt, MPI_DOUBLE, 0, 987, comm)); 

     }
     free(request_list);

     return 0;
}

int p2p_bcast_congestor(CommConfig_t *config, MPI_Comm comm, int myrank, int comm_ranks)
{
     int i;
     MPI_Request *request_list = NULL; 
     request_list = malloc(sizeof(MPI_Request) * comm_ranks);
     if (request_list == NULL) {
          die("Failed to allocate request_list in p2p_bcast_congestor()\n");
     }

     if(myrank == 0) {    
       
          for(i=1; i < comm_ranks; i++) {
               mpi_error(MPI_Isend(config->p2p_buffer, config->bcast_cnt, MPI_DOUBLE, i, 987, comm, &request_list[i-1])); 
          } 
          mpi_error(MPI_Waitall(comm_ranks-1, request_list, MPI_STATUS_IGNORE));

     } else {

          mpi_error(MPI_Recv(config->p2p_buffer, config->bcast_cnt, MPI_DOUBLE, 0, 987, comm, MPI_STATUS_IGNORE)); 

     }
     free(request_list);

     return 0;
}

int a2a_congestor(CommConfig_t *config, MPI_Comm comm, int myrank, int comm_ranks)
{
     int i, pof2, src, dst;
     i = 1;
  
     /* comm_size a power-of-two? */
     while (i < comm_ranks)
          i *= 2;
     if (i == comm_ranks)
          pof2 = 1;
     else
          pof2 = 0;

     /* do the pairwise exchanges */
     for(i = 0; i < comm_ranks; i++) {
 
          if (pof2 == 1) {
               /* use exclusive-or algorithm */
               src = dst = myrank ^ i;
          } else {
               src = (myrank - i + comm_ranks) % comm_ranks;
               dst = (myrank + i) % comm_ranks;
          }
    
          mpi_error(MPI_Sendrecv(&config->a2a_sbuffer[i * config->a2a_cnt], config->a2a_cnt, MPI_DOUBLE, 
                                 dst, 987, &config->a2a_rbuffer[i * config->a2a_cnt], config->a2a_cnt, MPI_DOUBLE, 
                                 src, 987, comm, MPI_STATUS_IGNORE));
     }

     return 0;
}

int rma_incast_congestor(CommConfig_t *config, MPI_Comm comm, int myrank, int comm_ranks)
{
     if (myrank != 0) {    
       
          mpi_error(MPI_Put(&config->rma_a2a_buffer[0], config->incast_cnt, MPI_DOUBLE, 0, 
                            (MPI_Aint)(myrank * config->incast_cnt), config->incast_cnt, MPI_DOUBLE, config->rma_a2a_window));
          mpi_error(MPI_Win_flush(0, config->rma_a2a_window));

     }

     return 0;
}

int rma_bcast_congestor(CommConfig_t *config, MPI_Comm comm, int myrank, int comm_ranks)
{
     if (myrank != 0) {    

          mpi_error(MPI_Get(&config->rma_buffer[0], config->bcast_cnt, MPI_DOUBLE, 0, 0, 
                            config->bcast_cnt, MPI_DOUBLE, config->rma_window));
          mpi_error(MPI_Win_flush(0, config->rma_window));

     }

     return 0;
}

int congestor(CommConfig_t *config, int n_measurements, int niters, MPI_Comm test_comm, CommTest_t req_test, 
              int record_perf, double * perfvals, double * perfval, int *real_n_measurements)
{
     int i, m, test_myrank, test_nranks;
     double bt1, bt2, bt;
     struct timespec t1, t2;
     MPI_Request req;
     double timeout_t1, timeout;

     mpi_error(MPI_Comm_rank(test_comm, &test_myrank));
     mpi_error(MPI_Comm_size(test_comm, &test_nranks));

     if (req_test == RMA_INCAST_CONGESTOR) {
          mpi_error(MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, config->rma_a2a_window));
     } else if (req_test == RMA_BCAST_CONGESTOR) {
          mpi_error(MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, config->rma_window));
     }

     timeout_t1 = MPI_Wtime();
     timeout    = 0.;
     mpi_error(MPI_Iallreduce(MPI_IN_PLACE, &timeout, 1, MPI_DOUBLE, MPI_MAX, test_comm, &req));
     *real_n_measurements = 0;
     bt = 0.;
     for (m = 0; m < n_measurements; m++) {

          /* check if we need to timeout this test because it is running too long */
          mpi_error(MPI_Wait(&req, MPI_STATUS_IGNORE));
          if (TEST_TIMEOUT_SECS <= timeout) continue;
          timeout = MPI_Wtime() - timeout_t1;
          mpi_error(MPI_Iallreduce(MPI_IN_PLACE, &timeout, 1, MPI_DOUBLE, MPI_MAX, test_comm, &req));

          if (record_perf) mpi_error(MPI_Barrier(test_comm)); 
          for (i = -1; i < niters; i++) {
               if (i == 0) bt1 = MPI_Wtime();
               if (i >= 0) clock_gettime(CLOCK_MONOTONIC, &t1);

               switch (req_test) {
               case A2A_CONGESTOR:
                    a2a_congestor(config, test_comm, test_myrank, test_nranks);
                    break;
               case P2P_INCAST_CONGESTOR:
                    p2p_incast_congestor(config, test_comm, test_myrank, test_nranks);
                    break;
               case RMA_INCAST_CONGESTOR:
                    rma_incast_congestor(config, test_comm, test_myrank, test_nranks);
                    break;
               case P2P_BCAST_CONGESTOR:
                    p2p_bcast_congestor(config, test_comm, test_myrank, test_nranks);
                    break;
               case RMA_BCAST_CONGESTOR:
                    rma_bcast_congestor(config, test_comm, test_myrank, test_nranks);
                    break;
               default:
                    break;
               }

               if (i >= 0 && record_perf) {
                    clock_gettime(CLOCK_MONOTONIC, &t2);
                    perfvals[m*niters + i] = 1e-9 * (double)(BILL * (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec));
               }

          }
          bt2 = MPI_Wtime();
          bt += bt2 - bt1;
          (*real_n_measurements)++;
     }
     if (req != MPI_REQUEST_NULL) {
          mpi_error(MPI_Wait(&req, MPI_STATUS_IGNORE));
     }

     if (record_perf) {

          if (req_test == A2A_CONGESTOR) {

               /* we report uni-directional BW in MiB/s/rank */
               for (i = 0; i < *real_n_measurements*niters; i++) {
                    perfvals[i] = (double)(sizeof(double) * config->a2a_cnt * (test_nranks-1)) / (perfvals[i] * 1024. * 1024.);
               }
               perfval[0] = (double)(sizeof(double) * config->a2a_cnt * *real_n_measurements*niters * 
                                     (test_nranks-1)) / (bt * 1024. * 1024.);

          } else if (req_test == P2P_INCAST_CONGESTOR || req_test == RMA_INCAST_CONGESTOR) {
      
               /* we report uni-directional BW in MiB/s/rank */
               if (test_myrank == 0) {
                    for (i = 0; i < *real_n_measurements*niters; i++) {
                         perfvals[i] = -1.0;
                    }
                    perfval[0] = -1.0;
               } else {
                    for (i = 0; i < *real_n_measurements*niters; i++) {
                         perfvals[i] = (double)(sizeof(double) * config->incast_cnt) / (perfvals[i] * 1024. * 1024.);
                    }
                    perfval[0] = (double)(sizeof(double) * config->incast_cnt * *real_n_measurements*niters) / 
                         (bt * 1024. * 1024.);
               }

          } else if (req_test == P2P_BCAST_CONGESTOR || req_test == RMA_BCAST_CONGESTOR) {
      
               /* we report uni-directional BW in MiB/s/rank */
               if (test_myrank == 0) {
                    for (i = 0; i < *real_n_measurements*niters; i++) {
                         perfvals[i] = -1.0;
                    }
                    perfval[0] = -1.0;
               } else {
                    for (i = 0; i < *real_n_measurements*niters; i++) {
                         perfvals[i] = (double)(sizeof(double) * config->bcast_cnt) / (perfvals[i] * 1024. * 1024.);
                    }
                    perfval[0] = (double)(sizeof(double) * config->bcast_cnt * *real_n_measurements*niters) / 
                         (bt * 1024. * 1024.);
               }

          }
     }

     if (req_test == RMA_INCAST_CONGESTOR) {
          mpi_error(MPI_Win_unlock(0, config->rma_a2a_window));
     } else if (req_test == RMA_BCAST_CONGESTOR) {
          mpi_error(MPI_Win_unlock(0, config->rma_window));
     }

     return 0;
}
