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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <network_test.h>

void die(char* errmsg)
{
     fprintf(stderr, "%s\n", errmsg);
     exit(EXIT_FAILURE);
}

void mpi_error(int ierr)
{
     if (ierr == MPI_SUCCESS) return;

     char err_string[MPI_MAX_ERROR_STRING];
     int err_length;

     MPI_Error_string(ierr, err_string, &err_length);
     fprintf(stderr, "%s\n", err_string);
     exit(1);
}

/* simple string sorting routine for ordering hostnames */
int cstring_cmp(const void *a, const void *b)
{
     const char **ia = (const char **)a;
     const char **ib = (const char **)b;
     return strcmp(*ia, *ib);
}

/* initialize MPI, detect node+rank layout, and setup various testing options  */
int init_mpi(CommConfig_t *config, CommNodes_t *nodes, int *argc, char ***argv, int rmacnt, int p2pcnt,
             int a2acnt, int incastcnt, int bcastcnt, int allreducecnt, int bw_outstanding)
{
     int i, ierr, nranks, hname_len;
     char local_hname[MPI_MAX_PROCESSOR_NAME], last_hname[MPI_MAX_PROCESSOR_NAME];
     char *all_hnames, **sort_all_hnames;

     config->bw_outstanding = bw_outstanding;
     config->p2pbw_cnt      = p2pcnt;
     config->rmabw_cnt      = rmacnt;
     config->a2a_cnt        = a2acnt;
     config->ar_cnt         = allreducecnt;
     config->incast_cnt     = incastcnt;
     config->bcast_cnt      = bcastcnt;
     config->rma_window     = MPI_WIN_NULL;
     config->rma_a2a_window = MPI_WIN_NULL;
     config->p2p_buffer     = NULL;
     config->a2a_sbuffer    = NULL;
     config->a2a_rbuffer    = NULL;
     config->ar_sbuffer     = NULL;
     config->ar_rbuffer     = NULL;

     mpi_error(MPI_Init(argc, argv));
     mpi_error(MPI_Comm_rank(MPI_COMM_WORLD, &config->myrank));
     mpi_error(MPI_Comm_size(MPI_COMM_WORLD, &config->nranks));

     /* get the list of nodes across all ranks */
     all_hnames      = malloc(sizeof(char) * config->nranks * MPI_MAX_PROCESSOR_NAME);
     if (all_hnames == NULL) {
          die("Failed to allocate all_hnames in init_mpi()\n");
     }
     sort_all_hnames = malloc(sizeof(char *) * config->nranks);
     if (sort_all_hnames == NULL) {
          die("Failed to allocate sort_all_hnames in init_mpi()\n");
     }
     for (i = 0; i < config->nranks; i++) {
          sort_all_hnames[i] = malloc(sizeof(char) * MPI_MAX_PROCESSOR_NAME);
          if (sort_all_hnames[i] == NULL) {
               die("Failed to allocate sort_all_hnames[] in init_mpi()\n");
          }
     }

     mpi_error(MPI_Get_processor_name(local_hname, &hname_len));
     mpi_error(MPI_Allgather(local_hname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_hnames,
                             MPI_MAX_PROCESSOR_NAME, MPI_CHAR, MPI_COMM_WORLD));

     for (i = 0; i < config->nranks; i++) {
          memcpy(sort_all_hnames[i], &all_hnames[i*MPI_MAX_PROCESSOR_NAME], MPI_MAX_PROCESSOR_NAME);
     }

     /* generate the list of unqiue nodes */
     qsort(sort_all_hnames, config->nranks, sizeof(char *), cstring_cmp);
     nodes->nnodes = 0;
     nodes->nodes_head = NULL;
     for (i = 0; i < config->nranks; i++) {
          ierr = strncmp(last_hname, sort_all_hnames[i], MPI_MAX_PROCESSOR_NAME);
          if (ierr != 0) {
               CommNode_t *tmp = malloc(sizeof(CommNode_t));
               if (tmp == NULL) {
                    die("Failed to allocate tmp CommNode_t in init_mpi()\n");
               }
               strcpy(tmp->host_name, sort_all_hnames[i]);
               tmp->ppn = 0;
               tmp->node_id = nodes->nnodes;
               tmp->next = NULL;
               tmp->last = NULL;
               tmp->ranks_head = NULL;
               if (nodes->nodes_head == NULL) {
                    nodes->nodes_head = tmp;
               } else {
                    tmp->last = nodes->nodes;
                    nodes->nodes->next = tmp;
               }
               nodes->nodes = tmp;
               nodes->nnodes++;
          }
          strcpy(last_hname, sort_all_hnames[i]);
     }

     /* scan back through and generate the ranks for each unique node */
     for (i = 0; i < config->nranks; i++) {
          CommNode_t *tmp = nodes->nodes_head;
          while (tmp != NULL) {
               ierr = strncmp(&all_hnames[i*MPI_MAX_PROCESSOR_NAME], tmp->host_name, MPI_MAX_PROCESSOR_NAME);
               if (ierr == 0) {
                    CommRank_t *rtmp = malloc(sizeof(CommRank_t));
                    if (tmp == NULL) {
                         die("Failed to allocate rtmp CommRank_t in init_mpi()\n");
                    }
                    rtmp->rank = i;
                    rtmp->next = NULL;
                    rtmp->last = NULL;
                    rtmp->node_rank = tmp->ppn;
                    if (rtmp->rank == config->myrank) {
                         config->mynode_rank = rtmp->node_rank;
                         config->mynode = tmp;
                    }
                    tmp->ppn++;
                    if (tmp->ranks_head == NULL) {
                         tmp->ranks_head = rtmp;
                    } else {
                         rtmp->last = tmp->ranks;
                         tmp->ranks->next = rtmp;
                    }
                    tmp->ranks = rtmp;
                    tmp = NULL;
               } else {
                    tmp = tmp->next;
               }
          }
     }

     for (i = 0; i < config->nranks; i++) {
          free(sort_all_hnames[i]);
     }
     free(sort_all_hnames);
     free(all_hnames);

     /* the factor of 4 is for 2 neighbors both send+recv */
     int count = 4 * p2pcnt;
     if (config->bcast_cnt > count) count = config->bcast_cnt;
     MPI_Aint p2p_length = sizeof(double) * (MPI_Aint)count;
     mpi_error(MPI_Alloc_mem(p2p_length, MPI_INFO_NULL, &config->p2p_buffer));
     if (ierr != MPI_SUCCESS) {
          die("Failed to allocate p2p_buffer in init_mpi()\n");
     }
     memset(config->p2p_buffer, 0, p2p_length);

     /* last we initialize the pretty table row separators */
     table_outerbar[TBLSIZE] = '\0';
     table_innerbar[TBLSIZE] = '\0';
     print_buffer[TBLSIZE]   = '\0';
     for (i = 0; i < TBLSIZE; i++) {
          table_outerbar[i] = '-';
          table_innerbar[i] = '-';
          print_buffer[i]   = '\0';
     }
     table_outerbar[0] = '+';
     table_outerbar[TBLSIZE-1] = '+';
     table_innerbar[0] = '+';
     table_innerbar[TBLSIZE-1] = '+';
     table_innerbar[34] = '+';
#ifdef VERBOSE
     for (i = 1; i < 7; i++) {
#else
     for (i = 1; i < 3; i++) {
#endif
          table_innerbar[34+i*15] = '+';
     }
     return 0;
}

int init_rma(CommConfig_t *config, MPI_Comm comm)
{
     int ierr;

     int count = 4 * config->rmabw_cnt;
     if (config->bcast_cnt > count) count = config->bcast_cnt;

     /* the factor of 4 is for 2 neighbors both send+recv */
     MPI_Aint window_length = (MPI_Aint)sizeof(double) * (MPI_Aint)count;
     ierr = MPI_Win_allocate(window_length, sizeof(double), MPI_INFO_NULL, comm, &config->rma_buffer, &config->rma_window);
     if (ierr != MPI_SUCCESS) {
          die("Failed to allocate RMA window in init_rma()\n");
     }

     memset(config->rma_buffer, 0, window_length);

     return 0;
}

int init_rma_a2a(CommConfig_t *config, MPI_Comm comm)
{
     int ierr, comm_size, count;

     count = config->a2a_cnt;
     if (config->incast_cnt > count) count = config->incast_cnt;

     mpi_error(MPI_Comm_size(comm, &comm_size));
     MPI_Aint window_length = (MPI_Aint)sizeof(double) * 2L * (MPI_Aint)count * (MPI_Aint)comm_size;
     ierr = MPI_Win_allocate(window_length, sizeof(double), MPI_INFO_NULL, comm, &config->rma_a2a_buffer, &config->rma_a2a_window);
     if (ierr != MPI_SUCCESS) {
          die("Failed to allocate RMA window in init_rma_a2a()\n");
     }

     memset(config->rma_a2a_buffer, 0, window_length);

     return 0;
}

/* cleanup data structures and finalize MPI */
int finalize_mpi(CommConfig_t *config, CommNodes_t *nodes)
{
     int i, j;

     CommNode_t *tmp = nodes->nodes;
     for (i = 0; i < nodes->nnodes; i++) {

          CommRank_t *rtmp = tmp->ranks;
          for (j = 0; j < tmp->ppn; j++) {
               CommRank_t *rprev = rtmp->last;
               free(rtmp);
               rtmp = rprev;
          }

          CommNode_t *prev = tmp->last;
          free(tmp);
          tmp = prev;
     }

     if (config->rma_window != MPI_WIN_NULL) {
          mpi_error(MPI_Win_free(&config->rma_window));
     }
     if (config->rma_a2a_window != MPI_WIN_NULL) {
          mpi_error(MPI_Win_free(&config->rma_a2a_window));
     }

     if (config->p2p_buffer != NULL ) MPI_Free_mem(config->p2p_buffer);
     if (config->a2a_sbuffer != NULL) MPI_Free_mem(config->a2a_sbuffer);
     if (config->a2a_rbuffer != NULL) MPI_Free_mem(config->a2a_rbuffer);
     if (config->ar_sbuffer != NULL ) MPI_Free_mem(config->ar_sbuffer);
     if (config->ar_rbuffer != NULL ) MPI_Free_mem(config->ar_rbuffer);
     mpi_error(MPI_Finalize());

     return 0;
}

/* shuffle a list in place */
void shuffle(int *list, int size, int seed, int call)
{
     int i, j, buff;
     if (call == 0) {
          srand(seed);
     }
     for (i = (size-1); i > 0; i--) {
          j = rand() % i;
          buff    = list[j];
          list[j] = list[i];
          list[i] = buff;
     }
}

/* separate allocation for A2A buffers since they are optional and large */
int a2a_buffers(CommConfig_t *config, MPI_Comm comm)
{
     int ierr, comm_size, count;

     count = config->a2a_cnt;
     if (config->incast_cnt > count) count = config->incast_cnt;

     mpi_error(MPI_Comm_size(comm, &comm_size));
     MPI_Aint length = sizeof(double) * count * comm_size;

     ierr = MPI_Alloc_mem(length, MPI_INFO_NULL, &config->a2a_sbuffer);
     if (ierr != MPI_SUCCESS) {
          die("Failed to allocate a2a_sbuffer in a2a_buffers()\n");
     }
     ierr = MPI_Alloc_mem(length, MPI_INFO_NULL, &config->a2a_rbuffer);
     if (ierr != MPI_SUCCESS) {
          die("Failed to allocate a2a_rbuffer in a2a_buffers()\n");
     }

     memset(config->a2a_sbuffer, 0, length);
     memset(config->a2a_rbuffer, 0, length);

     return 0;
}

/* separate allocation for allreduce buffers since they are optional and large */
int allreduce_buffers(CommConfig_t *config, MPI_Comm comm)
{
     int ierr, comm_size, count;

     count = config->ar_cnt;

     mpi_error(MPI_Comm_size(comm, &comm_size));
     MPI_Aint length = sizeof(double) * count;

     ierr = MPI_Alloc_mem(length, MPI_INFO_NULL, &config->ar_sbuffer);
     if (ierr != MPI_SUCCESS) {
          die("Failed to allocate a2a_sbuffer in a2a_buffers()\n");
     }
     ierr = MPI_Alloc_mem(length, MPI_INFO_NULL, &config->ar_rbuffer);
     if (ierr != MPI_SUCCESS) {
          die("Failed to allocate a2a_rbuffer in a2a_buffers()\n");
     }

     memset(config->ar_sbuffer, 0, length);
     memset(config->ar_rbuffer, 0, length);

     return 0;
}

int print_results(CommConfig_t *config, int localrank, int havedata, int from_min, char *name, char *units, CommResults_t * results)
{
     if (havedata && localrank == 0) {
          double avgworst = results->avgmin;
          if (from_min) avgworst = results->avgmax;
#ifdef VERBOSE
          snprintf(print_buffer, TBLSIZE+1, "| %31.31s | %12.1f | %12.1f | %12.1f | %12.1f | %12.1f | %12.1f | %12.12s |",
                   name, results->minval, results->maxval, results->avg, avgworst, results->percentile_99,
                   results->percentile_999, units);
#else
          snprintf(print_buffer, TBLSIZE+1, "| %31.31s | %12.1f | %12.1f | %12.12s |",
                   name, results->avg, results->percentile_99, units);
#endif
     }

     /* if we are not comm_world rank 0 and we have forward it to comm_world rank 0 */
     if (localrank == 0 && havedata && config->myrank != 0 ) {
          mpi_error(MPI_Send(print_buffer, TBLSIZE, MPI_CHAR, 0, 511, MPI_COMM_WORLD));
     } else if (config->myrank == 0 && (localrank != 0 || ! havedata)) {
          mpi_error(MPI_Recv(print_buffer, TBLSIZE, MPI_CHAR, MPI_ANY_SOURCE, 511, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
     }

     if (config->myrank == 0) {
          printf("%s\n", print_buffer);
          printf("%s\n", table_innerbar);
          fflush(stdout);
     }

     return 0;
}

int print_comparison_results(CommConfig_t *config, int localrank, int havedata, int from_min,
                             char *name, CommResults_t * base_results, CommResults_t * results)
{
     int fl=16;

     if (havedata && localrank == 0) {

          char b99[fl], bavg[fl], n99[fl], navg[fl];
          snprintf(n99, fl, "%12.1f", results->percentile_99);
          snprintf(navg, fl, "%12.1f", results->avg);
          snprintf(b99, fl, "%12.1f", base_results->percentile_99);
          snprintf(bavg, fl, "%12.1f", base_results->avg);

          double slowdown99  = (from_min) ? atof(n99) / atof(b99) : atof(b99) / atof(n99);
          double slowdownavg = (from_min) ? atof(navg) / atof(bavg) : atof(bavg) / atof(navg);
          snprintf(print_buffer, TBLSIZE+1, "| %31.31s | %19.1fX | %18.1fX |", name, slowdownavg, slowdown99);
     }

     /* if we are not comm_world rank 0 and we have forward it to comm_world rank 0 */
     if (localrank == 0 && havedata && config->myrank != 0 ) {
          mpi_error(MPI_Send(print_buffer, TBLSIZE, MPI_CHAR, 0, 512, MPI_COMM_WORLD));
     } else if (config->myrank == 0 && (localrank != 0 || ! havedata)) {
          mpi_error(MPI_Recv(print_buffer, TBLSIZE, MPI_CHAR, MPI_ANY_SOURCE, 512, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
     }

     if (config->myrank == 0) {
          printf("%s\n", print_buffer);
          printf("+---------------------------------+----------------------+---------------------+\n");
          fflush(stdout);
     }

     return 0;
}

int print_header(CommConfig_t *config, int type, CommTest_t ntype)
{
     int i;

     if (config->myrank == 0) {
          if (type < 3) printf("\n%s\n",table_outerbar);
          if (type == 0) {
#ifdef VERBOSE
               printf("| %57.57s%22.22s%57.57s |\n", " ", "Isolated Network Tests", " ");
#else
               printf("| %27.27s%22.22s%27.27s |\n", " ", "Isolated Network Tests", " ");
#endif
          } else if (type == 1) {
#ifdef VERBOSE
               printf("| %55.55s%25.25s%56.56s |\n", " ", "Isolated Congestion Tests", " ");
#else
               printf("| %25.25s%25.25s%26.26s |\n", " ", "Isolated Congestion Tests", " ");
#endif
          } else if (type == 2) {

               int nl=64;
               char nname[nl];
               for (i = 0; i < nl; i++) {
                    nname[i] = '\0';
               }
               switch (ntype) {
               case P2P_LATENCY:
                    snprintf(nname, nl, "RR Two-sided Lat");
                    break;
               case RMA_LATENCY:
                    snprintf(nname, nl, "RR Get Lat");
                    break;
               case P2P_BANDWIDTH:
                    snprintf(nname, nl, "RR Two-sided BW");
                    break;
               case RMA_BANDWIDTH:
                    snprintf(nname, nl, "RR Put BW");
                    break;
               case P2P_BANDWIDTH_NAT:
                    snprintf(nname, nl, "Nat Two-sided BW");
                    break;
               case P2P_NEIGHBORS:
                    snprintf(nname, nl, "RR Two-sided BW+Sync");
                    break;
               case ALLREDUCE_LATENCY:
                    snprintf(nname, nl, "Multiple Allreduce");
                    break;
               case A2A_BANDWIDTH:
                    snprintf(nname, nl, "Multiple Alltoall");
                    break;
               default:
                    break;
               }

#ifdef VERBOSE
               printf("| %28.28s%43.43s (%20s Network Test)%29s |\n", " ", "Network Tests running with Congestion Tests", nname, " ");
#else
               printf("| %16.16s%43.43s%17.17s |\n", " ", "Network Tests running with Congestion Tests", " ");
#endif
          } else if (type == 3) {

               printf("\n+------------------------------------------------------------------------------+\n");
               printf("| %9.9s%57.57s%10.10s |\n", " ", "Network Tests running with Congestion Tests - Key Results", " ");
               printf("+---------------------------------+--------------------------------------------+\n");
               printf("| %31.31s | %42.42s |\n",
                      "Name", "Congestion Impact Factor");
               printf("+---------------------------------+----------------------+---------------------+\n");
               printf("| %31.31s | %20.20s | %19.19s |\n",
                      " ", "Avg", "99%");
               printf("+---------------------------------+----------------------+---------------------+\n");
          }

          if (type < 3) {
               printf("%s\n",table_innerbar);
#ifdef VERBOSE
               printf("| %31.31s | %12.12s | %12.12s | %12.12s | %12.12s | %12.12s | %12.12s | %12.12s |\n",
                      "Name", "Min", "Max", "Avg", "Avg(Worst)", "99%", "99.9%", "Units");
#else
               printf("| %31.31s | %12.12s | %12.12s | %12.12s |\n",
                      "Name", "Avg", "99%", "Units");
#endif
               printf("%s\n",table_innerbar);
          }
          fflush(stdout);
     }

     return 0;
}

/* summarize performance stats */
int summarize_performance(CommConfig_t *config, double *myperf_vals_hires, double *myperf_vals,
                          int total_vals_hires, int total_vals, int from_min, MPI_Comm comm, CommResults_t *results)
{
     int i, j, myrank, nranks;
     double hires_min, hires_max, dres;
     uint64_t *distribution, myvalid_hires_vals, valid_hires_vals;

     mpi_error(MPI_Comm_rank(comm, &myrank));
     mpi_error(MPI_Comm_size(comm, &nranks));

     /* manage the high resolution data first */
     results->ndist_buckets = 100000;
     distribution = malloc(sizeof(double) * results->ndist_buckets);
     if (distribution == NULL) {
          die("Failed to allocate distribution in summarize_performance()\n");
     }
     for (j = 0; j < results->ndist_buckets; j++) distribution[j] = 0L;

     hires_min = 1.0e8;
     hires_max = -1.0e8;
     for (i = 0; i < total_vals_hires; i++) {
          hires_min = (hires_min > myperf_vals_hires[i] && myperf_vals_hires[i] >= 0.) ? myperf_vals_hires[i] : hires_min;
          hires_max = (hires_max < myperf_vals_hires[i] && myperf_vals_hires[i] >= 0.) ? myperf_vals_hires[i] : hires_max;
     }

     mpi_error(MPI_Allreduce(MPI_IN_PLACE, &hires_min, 1, MPI_DOUBLE, MPI_MIN, comm));
     mpi_error(MPI_Allreduce(MPI_IN_PLACE, &hires_max, 1, MPI_DOUBLE, MPI_MAX, comm));

     myvalid_hires_vals = 0L;
     dres = (hires_max - hires_min) / (double)(results->ndist_buckets - 1);
     for (i = 0; i < total_vals_hires; i++) {
          int bucket = (int)((myperf_vals_hires[i] - hires_min) / dres);
          if (bucket >= 0 && bucket < results->ndist_buckets) {
               distribution[bucket]++;
               myvalid_hires_vals++;
          }
     }

     if (myrank == 0) {
          if (results->distribution != NULL) free(results->distribution);
          results->distribution = malloc(sizeof(uint64_t) * results->ndist_buckets);
          if (results->distribution == NULL) {
               die("Failed to allocate results->distribution in summarize_performance()\n");
          }
          for (j = 0; j < results->ndist_buckets; j++) {
               results->distribution[j] = 0L;
          }
     }
     mpi_error(MPI_Reduce(distribution, results->distribution, results->ndist_buckets, MPI_UINT64_T, MPI_SUM, 0, comm));
     free(distribution);
     mpi_error(MPI_Reduce(&myvalid_hires_vals, &valid_hires_vals, 1, MPI_UINT64_T, MPI_SUM, 0, comm));

     if (myrank == 0) {
          uint64_t cnt_99 = (uint64_t)(0.99 * (double)valid_hires_vals);
          uint64_t cnt_999 = (uint64_t)(0.999 * (double)valid_hires_vals);
          int f99 = 0, f999 = 0;
          uint64_t cnt = 0L;

          if (from_min == 1) {
               for (j = 0; j < results->ndist_buckets; j++) {
                    cnt += results->distribution[j];

                    if (cnt >= cnt_99 && f99 == 0) {
                         results->percentile_99 = hires_min + (double)(j+1)*dres;
                         f99 = 1;
                    }

                    if (cnt >= cnt_999 && f999 == 0) {
                         results->percentile_999 = hires_min + (double)(j+1)*dres;
                         f999 = 1;
                    }
               }
          } else {
               for (j = results->ndist_buckets-1; j >=0; j--) {
                    cnt += results->distribution[j];

                    if (cnt >= cnt_99 && f99 == 0) {
                         results->percentile_99 = hires_min + (double)j*dres;
                         f99 = 1;
                    }

                    if (cnt >= cnt_999 && f999 == 0) {
                         results->percentile_999 = hires_min + (double)j*dres;
                         f999 = 1;
                    }
               }
          }
          results->dlow = hires_min;
          results->dhi  = hires_max;
          results->dres = dres;
     }

     /* now do the data averaged over iterations */
     double *minperf_vals, *maxperf_vals, *avgperf_vals;
     if (myrank == 0) {

          minperf_vals = malloc(sizeof(double) * total_vals);
          maxperf_vals = malloc(sizeof(double) * total_vals);
          avgperf_vals = malloc(sizeof(double) * total_vals);

          if (minperf_vals == NULL || maxperf_vals == NULL || avgperf_vals == NULL) {
               die("Failed to allocate global metric arrays in summarize_performance()\n");
          }

     }

     /* generate the cross-rank statistics. handle the case where our own data is not
        of use likely because we are a root rank that did nothing. */
     double *filt_perf_vals = malloc(sizeof(double) * total_vals);
     int *have_valid_val    = malloc(sizeof(int) * total_vals);
     int *valid_vals        = malloc(sizeof(int) * total_vals);

     if (filt_perf_vals == NULL || have_valid_val == NULL || valid_vals == NULL) {
          die("Failed to allocate metric arrays in summarize_performance()\n");
     }

     for (i = 0; i < total_vals; i++) have_valid_val[i] = (myperf_vals[i] > 0.) ? 1 : 0;
     mpi_error(MPI_Reduce(have_valid_val, valid_vals, total_vals, MPI_INT, MPI_SUM, 0, comm));

     for (i = 0; i < total_vals; i++) filt_perf_vals[i] = (myperf_vals[i] > 0.) ? myperf_vals[i] : 1.0e8;
     mpi_error(MPI_Reduce(filt_perf_vals, minperf_vals, total_vals, MPI_DOUBLE, MPI_MIN, 0, comm));

     for (i = 0; i < total_vals; i++) filt_perf_vals[i] = (myperf_vals[i] > 0.) ? myperf_vals[i] : -1.0e8;
     mpi_error(MPI_Reduce(filt_perf_vals, maxperf_vals, total_vals, MPI_DOUBLE, MPI_MAX, 0, comm));

     for (i = 0; i < total_vals; i++) filt_perf_vals[i] = (myperf_vals[i] > 0.) ? myperf_vals[i] : 0.;
     mpi_error(MPI_Reduce(filt_perf_vals, avgperf_vals, total_vals, MPI_DOUBLE, MPI_SUM, 0, comm));

     if (myrank == 0) {

          for (i = 0; i < total_vals; i++) {
               avgperf_vals[i] = avgperf_vals[i] / (double)valid_vals[i];
          }

          /* now compute statistics looking across the total measurments */
          results->minval = 1.0e8;
          results->maxval = -1.0e8;
          results->avg    = 0.;
          results->avgmax = 0.;
          results->avgmin = 0.;
          for (i = 0; i < total_vals; i++) {
               results->minval  = (minperf_vals[i] < results->minval) ? minperf_vals[i] : results->minval;
               results->maxval  = (maxperf_vals[i] > results->maxval) ? maxperf_vals[i] : results->maxval;
               results->avg    += avgperf_vals[i];
               results->avgmax += maxperf_vals[i];
               results->avgmin += minperf_vals[i];
          }

          results->avg    = results->avg / total_vals;
          results->avgmax = results->avgmax / total_vals;
          results->avgmin = results->avgmin / total_vals;
     }

     free(filt_perf_vals);
     free(have_valid_val);
     free(valid_vals);
     if (myrank == 0) {
          free(minperf_vals);
          free(maxperf_vals);
          free(avgperf_vals);
     }

     return 0;
}

int create_perf_filename(CommTest_t req_test, CommTest_t other_test, int isbaseline, char *suffix, char **fname)
{
     int i, bl=16, ol=64, fl=256;
     char bases[bl], othern[ol];

     *fname = malloc(sizeof(char) * fl);
     if (*fname == NULL) {
          die("Failed to allocate fname in create_perf_filename()\n");
     }

     for (i = 0; i < bl; i++) {
          bases[i] = '\0';
     }

     if (isbaseline) snprintf(bases, bl, "_baseline");

     for (i = 0; i < ol; i++) {
          othern[i] = '\0';
     }

     switch (other_test) {
     case P2P_LATENCY:
          snprintf(othern, ol, "_with_p2p_latency");
          break;
     case RMA_LATENCY:
          snprintf(othern, ol, "_with_get_latency");
          break;
     case P2P_BANDWIDTH:
          snprintf(othern, ol, "_with_p2p_bandwidth");
          break;
     case P2P_BANDWIDTH_NAT:
          snprintf(othern, ol, "_with_p2p_bandwidth_nat");
          break;
     case RMA_BANDWIDTH:
          snprintf(othern, ol, "_with_put_bandwidth");
          break;
     case P2P_NEIGHBORS:
          snprintf(othern, ol, "_with_p2p_neighbors");
          break;
     case ALLREDUCE_LATENCY:
          snprintf(othern, ol, "_with_allreduce_latency");
          break;
     case A2A_BANDWIDTH:
          snprintf(othern, ol, "_with_a2a_bandwidth");
          break;
     case A2A_CONGESTOR:
          snprintf(othern, ol, "_with_a2a_congestor");
          break;
     case ALLREDUCE_CONGESTOR:
          snprintf(othern, ol, "_with_allreduce_congestor");
          break;
     case P2P_INCAST_CONGESTOR:
          snprintf(othern, ol, "_with_p2p_incast_congestor");
          break;
     case P2P_BCAST_CONGESTOR:
          snprintf(othern, ol, "_with_p2p_bcast_congestor");
          break;
     case RMA_INCAST_CONGESTOR:
          snprintf(othern, ol, "_with_put_incast_congestor");
          break;
     case RMA_BCAST_CONGESTOR:
          snprintf(othern, ol, "_with_get_bcast_congestor");
          break;
     case TEST_CONGESTORS:
          snprintf(othern, ol, "_with_congestors");
          break;
     default:
          break;
     }

     switch (req_test) {
     case P2P_LATENCY:
          snprintf(*fname, fl, "p2p_latency%s%s.%s", bases, othern, suffix);
          break;
     case RMA_LATENCY:
          snprintf(*fname, fl, "get_latency%s%s.%s", bases, othern, suffix);
          break;
     case P2P_BANDWIDTH:
          snprintf(*fname, fl, "p2p_bandwidth%s%s.%s", bases, othern, suffix);
          break;
     case P2P_BANDWIDTH_NAT:
          snprintf(*fname, fl, "p2p_bandwidth_nat%s%s.%s", bases, othern, suffix);
          break;
     case RMA_BANDWIDTH:
          snprintf(*fname, fl, "put_bandwidth%s%s.%s", bases, othern, suffix);
          break;
     case P2P_NEIGHBORS:
          snprintf(*fname, fl, "p2p_neighbors%s%s.%s", bases, othern, suffix);
          break;
     case ALLREDUCE_LATENCY:
          snprintf(*fname, fl, "allreduce_latency%s%s.%s", bases, othern, suffix);
          break;
     case A2A_BANDWIDTH:
          snprintf(*fname, fl, "a2a_bandwidth%s%s.%s", bases, othern, suffix);
          break;
     case A2A_CONGESTOR:
          snprintf(*fname, fl, "a2a_congestor%s%s.%s", bases, othern, suffix);
          break;
     case ALLREDUCE_CONGESTOR:
          snprintf(*fname, fl, "allreduce_congestor%s%s.%s", bases, othern, suffix);
          break;
     case P2P_INCAST_CONGESTOR:
          snprintf(*fname, fl, "p2p_incast_congestor%s%s.%s", bases, othern, suffix);
          break;
     case P2P_BCAST_CONGESTOR:
          snprintf(*fname, fl, "p2p_bcast_congestor%s%s.%s", bases, othern, suffix);
          break;
     case RMA_INCAST_CONGESTOR:
          snprintf(*fname, fl, "put_incast_congestor%s%s.%s", bases, othern, suffix);
          break;
     case RMA_BCAST_CONGESTOR:
          snprintf(*fname, fl, "get_bcast_congestor%s%s.%s", bases, othern, suffix);
          break;
     default:
          break;
     }

     return 0;
}

/* create a string record of the performance a rank sees on a test */
int summarize_pairs_performance(CommConfig_t *config, MPI_Comm comm, char *lnode, char *rnode, double *myperf_vals,
                                int nsamps, int m, int r, CommTest_t req_test, CommTest_t other_test)
{
     double minv, maxv, avgv;
     int i, myrank, nranks, tl=64, pl=256;
     char tname[tl], tunits[tl], stime[tl], perf_str[pl];
     char *all_perf_strs;
     struct timespec timestmp;

#ifndef VERBOSE
     return 0;
#else

     mpi_error(MPI_Comm_size(comm, &nranks));
     mpi_error(MPI_Comm_rank(comm, &myrank));

     if (myrank == 0) {
          all_perf_strs = malloc(sizeof(char) * pl * nranks);
          if (all_perf_strs == NULL) {
               die("Root rank failed to allocate all_perf_strs in random_ring()\n");
          }
     }

     for (i = 0; i < tl; i++) {
          tname[i]  = '\0';
          tunits[i] = '\0';
     }
     for (i = 0; i < pl; i++) {
          perf_str[i] = '\0';
     }

     switch (req_test) {
     case P2P_LATENCY:
          snprintf(tname, tl, "p2p_latency");
          snprintf(tunits, tl, "usec");
          break;
     case RMA_LATENCY:
          snprintf(tname, tl, "get_latency");
          snprintf(tunits, tl, "usec");
          break;
     case P2P_BANDWIDTH:
          snprintf(tname, tl, "p2p_bandwidth");
          snprintf(tunits, tl, "MiB/s");
          break;
     case P2P_BANDWIDTH_NAT:
          snprintf(tname, tl, "p2p_bandwidth_nat");
          snprintf(tunits, tl, "MiB/s");
          break;
     case RMA_BANDWIDTH:
          snprintf(tname, tl, "put_bandwidth");
          snprintf(tunits, tl, "MiB/s");
          break;
     case P2P_NEIGHBORS:
          snprintf(tname, tl, "p2p_neighbors");
          snprintf(tunits, tl, "MiB/s");
          break;
     case ALLREDUCE_LATENCY:
          snprintf(tname, tl, "allreduce_latency");
          snprintf(tunits, tl, "usec");
          break;
     case A2A_BANDWIDTH:
          snprintf(tname, tl, "a2a_bandwidth");
          snprintf(tunits, tl, "MiB/s");
          break;
     case A2A_CONGESTOR:
          snprintf(tname, tl, "a2a_congestor");
          snprintf(tunits, tl, "MiB/s");
          break;
     case P2P_INCAST_CONGESTOR:
          snprintf(tname, tl, "p2p_incast_congestor");
          snprintf(tunits, tl, "MiB/s");
          break;
     case P2P_BCAST_CONGESTOR:
          snprintf(tname, tl, "p2p_bcast_congestor");
          snprintf(tunits, tl, "MiB/s");
          break;
     case RMA_INCAST_CONGESTOR:
          snprintf(tname, tl, "put_incast_congestor");
          snprintf(tunits, tl, "MiB/s");
          break;
     case RMA_BCAST_CONGESTOR:
          snprintf(tname, tl, "get_bcast_congestor");
          snprintf(tunits, tl, "MiB/s");
          break;
     default:
          break;
     }

     minv = 1e8;
     maxv = -1e8;
     avgv = 0.0;

     for (i = 0; i < nsamps; i++) {

          minv = (myperf_vals[i] < minv) ? myperf_vals[i] : minv;
          maxv = (myperf_vals[i] > maxv) ? myperf_vals[i] : maxv;
          avgv += myperf_vals[i];

     }
     avgv /= nsamps;

     /* string is
        datetime reporting_global_rank  reporting_local_rank  reporting_node  left_remote_node  right_remote_node  meas_idx  rand_idx  test_name  units  #samps  min  max  avg
     */
     clock_gettime(CLOCK_REALTIME, &timestmp);
     snprintf(stime, tl, "%19.4f", 1e-9 * (1000000000L * timestmp.tv_sec + timestmp.tv_nsec));
     snprintf(perf_str, pl, "%20.20s    %8i    %8i    %20.20s    %20.20s    %20.20s    %6i    %6i    %10s    %10s    %8i    %20.5f    %20.5f    %20.5f\n",
              stime, config->myrank, config->mynode_rank, config->mynode->host_name, lnode, rnode, m, r, tname, tunits, nsamps, minv, maxv, avgv);

     /* gather all pairs perf results to global_comm rank 0 and write them to a file */
     mpi_error(MPI_Gather(perf_str, pl, MPI_CHAR, all_perf_strs, pl, MPI_CHAR, 0, comm));
     if (myrank == 0) {

          FILE *fp;
          char *fname;

          int isbaseline = (other_test == TEST_NULL) ? 1 : 0;
          create_perf_filename(req_test, other_test, isbaseline, "rec", &fname);
          fp = fopen(fname, "a+");
          for (i = 0; i < nranks; i++) {
               fprintf(fp, "%256.256s\n", &all_perf_strs[i*pl]);
          }
          fclose(fp);
          free(fname);

     }

     if (myrank == 0) {
          free(all_perf_strs);
     }

     return 0;
#endif
}

/* create a file with the distribution of hires measurements */
int write_distribution(CommTest_t req_test, CommTest_t other_test, int isbaseline,
                       CommResults_t * results, char * tname, char * tunits)
{
     int i;
     FILE *fp;
     char *fname;

#ifndef VERBOSE
     return 0;
#else

     create_perf_filename(req_test, other_test, isbaseline, "dat", &fname);

     fp = fopen(fname, "w+");
     fprintf(fp, "%22.22s%20.20s\n", " ", tname);
     fprintf(fp, "%20.20s %20.20s %20.20s\n", "Bin", tunits, "count");
     for (i = 0; i < results->ndist_buckets; i++) {
          fprintf(fp, "%20i %20.5f %20lu\n", i, (results->dlow+i*results->dres), results->distribution[i]);
     }
     fclose(fp);
     free(fname);

     return 0;
#endif
}
