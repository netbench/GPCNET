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

#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <network_test.h>

#define VERSION 1.1

#define NUM_NETWORK_TESTS 8

/* loop counts for the various tests */
#define NUM_LAT_TESTS 10000
#define NUM_LAT_RANDS 30
#define NUM_LAT_ITERS 200
#define NUM_BW_TESTS 10000
#define NUM_BW_RANDS 30
#define NUM_BW_ITERS 8
#define NUM_ALLREDUCE_TESTS 100000
#define NUM_ALLREDUCE_ITERS 200
#define NUM_A2A_TESTS 10000
#define NUM_A2A_ITERS 64

/* test specific tuning */
#define BW_MSG_COUNT 16384
#define BW_OUTSTANDING 8
#define A2A_MSG_COUNT 512

CommTest_t network_tests_list[NUM_NETWORK_TESTS];

int network_test_setup(CommTest_t req_test, int *ntests, int *nrands, int *niters, 
                       char *tname, char *tunits)
{
     int nl=64;

     switch (req_test) {
     case P2P_LATENCY:
          *ntests = NUM_LAT_TESTS;
          *nrands = NUM_LAT_RANDS;
          *niters = NUM_LAT_ITERS;
          snprintf(tname, nl, "%s (8 B)", "RR Two-sided Lat");
          snprintf(tunits, nl, "%s", "usec");
          break;
     case RMA_LATENCY:
          *ntests = NUM_LAT_TESTS;
          *nrands = NUM_LAT_RANDS;
          *niters = NUM_LAT_ITERS;
          snprintf(tname, nl, "%s (8 B)", "RR Get Lat");
          snprintf(tunits, nl, "%s", "usec");
          break;
     case P2P_BANDWIDTH:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "RR Two-sided BW", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case P2P_BANDWIDTH_NAT:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "Nat Two-sided BW", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case RMA_BANDWIDTH:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "RR Put BW", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case P2P_NEIGHBORS:
          *ntests = NUM_BW_TESTS;
          *nrands = NUM_BW_RANDS;
          *niters = NUM_BW_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "RR Two-sided BW+Sync", (int)(sizeof(double)*BW_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     case ALLREDUCE_LATENCY:
          *ntests = NUM_ALLREDUCE_TESTS;
          *nrands = 1;
          *niters = NUM_ALLREDUCE_ITERS;
          snprintf(tname, nl, "%s (8 B)", "Multiple Allreduce");
          snprintf(tunits, nl, "%s", "usec");
          break;
     case A2A_BANDWIDTH:
          *ntests = NUM_A2A_TESTS;
          *nrands = 1;
          *niters = NUM_A2A_ITERS;
          snprintf(tname, nl, "%s (%4i B)", "Multiple Alltoall", (int)(sizeof(double)*A2A_MSG_COUNT));
          snprintf(tunits, nl, "%s", "MiB/s/rank");
          break;
     default:
          break;
     }

     return 0;
}

int main(int argc, char* argv[])
{
     CommConfig_t test_config;
     CommNodes_t nodes;
     CommResults_t results;
     MPI_Comm test_comm;
     int nl=64;
     char tname[nl], tunits[nl];
     int itest, niters, ntests, nrands, i;
     int *allnodes;

     init_mpi(&test_config, &nodes, &argc, &argv, BW_MSG_COUNT, BW_MSG_COUNT, A2A_MSG_COUNT, 
              1, 1, BW_OUTSTANDING);

     if (nodes.nnodes < 2) {
          if (test_config.myrank == 0) {
               printf("ERROR: this application must be run on at least 2 nodes\n");
          }
          MPI_Finalize();
          exit(1);
     }

     /* use subcommunicators that are only off node for each rank */
     allnodes = malloc(sizeof(int) * nodes.nnodes);
     for (i = 0; i < nodes.nnodes; i++) {
          allnodes[i] = i;
     }
     node_slice_subcomms(&test_config, &nodes, allnodes, nodes.nnodes, &test_comm);
     free(allnodes);

     init_rma(&test_config, test_comm);

     /* set the order of tests to run */
     network_tests_list[0] = P2P_LATENCY;
     network_tests_list[1] = RMA_LATENCY;
     network_tests_list[2] = P2P_BANDWIDTH;
     network_tests_list[3] = RMA_BANDWIDTH;
     network_tests_list[4] = P2P_NEIGHBORS;
     network_tests_list[5] = P2P_BANDWIDTH_NAT;
     network_tests_list[6] = ALLREDUCE_LATENCY;
     network_tests_list[7] = A2A_BANDWIDTH;

     if (test_config.myrank == 0) {
          printf("Network Tests v%3.1f\n", VERSION);
          printf("  Test with %i MPI ranks (%i nodes)\n\n", test_config.nranks, nodes.nnodes);
          printf("  Legend\n   RR = random ring communication pattern\n   Nat = natural ring communication pattern\n   Lat = latency\n   BW = bandwidth\n   BW+Sync = bandwidth with barrier");
     }
    
     /* gather the baseline performance */
     results.distribution = NULL;
     print_header(&test_config, 0, 0);
     for (itest = 0; itest < NUM_NETWORK_TESTS; itest++) {

          network_test_setup(network_tests_list[itest], &ntests, &nrands, &niters, tname, tunits);
          if (network_tests_list[itest] != A2A_BANDWIDTH && network_tests_list[itest] 
              != P2P_BANDWIDTH_NAT && network_tests_list[itest] != ALLREDUCE_LATENCY) {
               random_ring(&test_config, 0, ntests, nrands, niters, network_tests_list[itest], 
                           TEST_NULL, test_comm, MPI_COMM_WORLD, &results);
          } else if (network_tests_list[itest] == P2P_BANDWIDTH_NAT) {
               random_ring(&test_config, 1, ntests, nrands, niters, P2P_BANDWIDTH, TEST_NULL, 
                           test_comm, MPI_COMM_WORLD, &results);
          } else if (network_tests_list[itest] == A2A_BANDWIDTH) {
               a2a_test(&test_config, ntests, niters, test_comm, MPI_COMM_WORLD, &results);
          } else if (network_tests_list[itest] == ALLREDUCE_LATENCY) {
               allreduce_test(&test_config, ntests, niters, test_comm, MPI_COMM_WORLD, &results);
          }
    
          int from_min = 0;
          if (network_tests_list[itest] == P2P_LATENCY || network_tests_list[itest] == RMA_LATENCY ||
              network_tests_list[itest] == ALLREDUCE_LATENCY) from_min = 1;
          print_results(&test_config, test_config.myrank, 1, from_min, tname, tunits, &results);

          if (test_config.myrank == 0) {
               write_distribution(network_tests_list[itest], TEST_NULL, 1, &results, tname, tunits);
          }

     }

     finalize_mpi(&test_config, &nodes);

     return 0;
}
