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

#ifndef HAVE_NETWORK_TEST_H
#define HAVE_NETWORK_TEST_H

#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define RSEED 8675309

/* these establish the size of the table written to STDOUT */
#ifdef VERBOSE
#define TBLSIZE 140
#else
#define TBLSIZE 80
#endif
char table_outerbar[TBLSIZE+1], table_innerbar[TBLSIZE+1], print_buffer[TBLSIZE+1];

/* define a timeout for how long a test can run */
#define TEST_TIMEOUT_SECS 10.0

typedef struct CommRank_st {
     int rank;
     int node_rank;
     struct CommRank_st *next;
     struct CommRank_st *last;
} CommRank_t;

typedef struct CommNode_st {
     char host_name[MPI_MAX_PROCESSOR_NAME];
     CommRank_t *ranks;
     CommRank_t *ranks_head;
     int ppn;
     int node_id;
     struct CommNode_st *next;
     struct CommNode_st *last;
} CommNode_t;

typedef struct CommNodes_st {
     CommNode_t *nodes;
     CommNode_t *nodes_head;
     int nnodes;
} CommNodes_t;

typedef struct CommConfig_st {
     struct CommNode_st *mynode;
     double *rma_buffer;
     double *rma_a2a_buffer;
     double *p2p_buffer;
     double *a2a_sbuffer;
     double *a2a_rbuffer;
     MPI_Win rma_window;
     MPI_Win rma_a2a_window;
     int bw_outstanding;
     int p2pbw_cnt;
     int rmabw_cnt;
     int a2a_cnt;
     int incast_cnt;
     int bcast_cnt;
     int myrank;
     int nranks;
     int mynode_rank;
} CommConfig_t;

typedef enum CommTest_st
{
     P2P_LATENCY = 0,
     P2P_BANDWIDTH,
     P2P_BANDWIDTH_NAT,
     RMA_BANDWIDTH,
     RMA_LATENCY,
     P2P_NEIGHBORS,
     ALLREDUCE_LATENCY,
     A2A_BANDWIDTH,
     A2A_CONGESTOR,
     P2P_INCAST_CONGESTOR,
     P2P_BCAST_CONGESTOR,
     RMA_INCAST_CONGESTOR,
     RMA_BCAST_CONGESTOR,
     TEST_CONGESTORS,
     TEST_NULL
} CommTest_t;

typedef struct CommResults_st {
     double minval;
     double maxval;
     double avg;
     double avgmax;
     double avgmin;
     double percentile_99;
     double percentile_999;
     uint64_t *distribution;
     double dlow;
     double dhi;
     double dres;
     int ndist_buckets;
} CommResults_t;

/* random_ring.c */
int random_ring(CommConfig_t *config, int norand, int n_measurements, int nrands, int niters, CommTest_t req_test, CommTest_t other_test,
                MPI_Comm comm, MPI_Comm global_comm, CommResults_t *results);
int finalize_mpi(CommConfig_t *config, CommNodes_t *nodes);
int p2p_latency(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, MPI_Comm comm, double *perfvals, double *perfval);
int p2p_bandwidth(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, MPI_Comm comm, double *perfvals, double *perfval);
int rma_latency(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, MPI_Comm comm, double *perfvals, double *perfval);
int rma_bandwidth(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, MPI_Comm comm, double *perfvals, double *perfval);
int p2p_neighbors(CommConfig_t *config, int lneighbor, int rneighbor, int niters, MPI_Comm global_comm, MPI_Comm comm, double *perfvals, double *perfval);

/* collectives.c */
int a2a_test(CommConfig_t *config, int ntests, int base_niters, MPI_Comm a2acomm, MPI_Comm global_comm, CommResults_t * results);
int allreduce_test(CommConfig_t *config, int ntests, int niters, MPI_Comm comm, MPI_Comm global_comm, CommResults_t * results);

/* subcomms.c */
int split_subcomms(int nsubcomms, MPI_Comm local_comm, MPI_Comm base_comm, int *color, MPI_Comm *test_comm, MPI_Comm *subcomm);
int node_slice_subcomms(CommConfig_t *config, CommNodes_t *nodes, int *node_list, int list_size, MPI_Comm *subcomm);
int congestion_subcomms(CommConfig_t *config, CommNodes_t *nodes, int *congestor_node_list, int list_size, int *am_congestor, MPI_Comm *subcomm);

/* utils.c */
void die(char *errmsg);
void mpi_error(int ierr);
int init_mpi(CommConfig_t *config, CommNodes_t *nodes, int *argc, char ***argv, int rmacnt, int p2pcnt, int a2acnt, int incastcnt, 
             int bcastcnt, int bw_outstanding);
int init_rma(CommConfig_t *config, MPI_Comm comm);
int init_rma_a2a(CommConfig_t *config, MPI_Comm comm);
int a2a_buffers(CommConfig_t *config, MPI_Comm subcomm);
void shuffle(int *list, int size, int seed, int call);
int print_results(CommConfig_t *config, int localrank, int havedata, int from_min, char *name, char *units, CommResults_t * results);
int print_comparison_results(CommConfig_t *config, int localrank, int havedata, int from_min,
                             char *name, CommResults_t * base_results, CommResults_t * results);
int print_header(CommConfig_t *config, int type, CommTest_t ctype);
int summarize_performance(CommConfig_t *config, double *myperf_vals_hires, double *myperf_vals, int total_vals_hires, int total_vals, 
                          int from_min, MPI_Comm comm, CommResults_t *results);
int write_distribution(CommTest_t req_test, CommTest_t other_test, int isbaseline, CommResults_t * results, char * tname, char * tunits);
int summarize_pairs_performance(CommConfig_t *config, MPI_Comm comm, char *lnode, char *rnode, double *myperf_vals, int nsamps, int m, int r, 
                                CommTest_t req_test, CommTest_t other_test);

/* congestors.c */
int congestor(CommConfig_t *config, int n_measurements, int niters, MPI_Comm test_comm, CommTest_t req_test, 
              int record_perf, double * perfvals, double * perfval, int *real_n_measurements);
int p2p_incast_congestor(CommConfig_t *config, MPI_Comm comm, int myrank, int comm_ranks);
int a2a_congestor(CommConfig_t *config, MPI_Comm comm, int myrank, int comm_ranks);

#endif
