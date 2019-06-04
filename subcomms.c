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

/* create a subcomm from the given one divided in the given number of segments randomly selected
   test_comm is a communicator that includes all ranks that will have the same color
   subcomm is the same but is intended to be generated after the node_slice comm */
int split_subcomms(int nsubcomms, MPI_Comm local_comm, MPI_Comm base_comm, int *color, 
                   MPI_Comm *test_comm, MPI_Comm *subcomm)
{
     int nranks_base, myrank_base, i;
     int *colors;
     
     mpi_error(MPI_Comm_size(base_comm, &nranks_base));
     mpi_error(MPI_Comm_rank(base_comm, &myrank_base));
     
     colors = malloc(sizeof(int) * nranks_base);
     if (colors == NULL) {
          die("Failed to allocate colors in split_subcomms()\n");
     }
     for (i = 0; i < nranks_base; i++) {
          colors[i] = i;
     }
    
     /* we'll use a shuffled rank list to randomize the color each rank is given */
     shuffle(colors, nranks_base, RSEED, 1);
     for (i = 0; i < nranks_base; i++) {
          colors[i] = colors[i] % nsubcomms;
     }
     
     mpi_error(MPI_Comm_split(base_comm, colors[myrank_base], 0, subcomm));
     mpi_error(MPI_Comm_split(local_comm, colors[myrank_base], 0, test_comm));
     *color = colors[myrank_base];
     free(colors);
     
     return 0;
}

/* create a subcomm that includes the same local rank as the calling rank on every other node */
int node_slice_subcomms(CommConfig_t *config, CommNodes_t *nodes, int *node_list, 
                        int list_size, MPI_Comm *subcomm)
{
     int i, j, k, group_size, ingroup;
     int *group_list;
     MPI_Group base_group;
     MPI_Group comm_group;
     MPI_Comm null_comm;
     CommNode_t **used_nodes;

     mpi_error(MPI_Comm_group(MPI_COMM_WORLD, &base_group));

     /* find the maximum ppn across the nodes we want and that is how many comm groups we'll make */
     used_nodes = malloc(sizeof(CommNode_t *) * list_size);
     if (used_nodes == NULL) {
          die("Failed to allocate used_nodes in node_slice_subcomms()\n");
     }
     int ngroups = 0;
     k = 0;
     for (i = 0; i < list_size; i++) {
          CommNode_t *tmp = nodes->nodes_head;
          for (j = 0; j < nodes->nnodes; j++) {
               if (tmp->node_id == node_list[i]) {
                    ngroups = (ngroups < tmp->ppn) ? tmp->ppn : ngroups;
                    used_nodes[k] = tmp;
                    k++;
               }
               tmp = tmp->next;
          }
     }

     for (i = 0; i < ngroups; i++) {

          /* determine the number of ranks each group will have */
          group_size = 0;
          for (j = 0; j < list_size; j++) {
               if (used_nodes[j]->ppn > i) group_size++;
          }
          group_list = malloc(sizeof(int) * group_size);
          if (group_list == NULL) {
               die("Failed to allocate group_list in node_slice_subcomms()\n");
          }

          /* determine the specific ranks one in this group */
          int grank = 0;
          ingroup = 0;
          for (j = 0; j < list_size; j++) {
               if (used_nodes[j]->ppn > i) {
                    CommRank_t *rtmp = used_nodes[j]->ranks_head;
                    for (k = 0; k < i; k++) {
                         rtmp = rtmp->next;
                    }
                    group_list[grank] = rtmp->rank;
                    if (config->myrank == group_list[grank]) ingroup = 1;
                    grank++;
               }
          }
          mpi_error(MPI_Group_incl(base_group, group_size, group_list,
                                   &comm_group));

          if (ingroup == 1) {
               mpi_error(MPI_Comm_create(MPI_COMM_WORLD, comm_group, subcomm));
          } else {
               mpi_error(MPI_Comm_create(MPI_COMM_WORLD, comm_group, &null_comm));
          }

          free(group_list);
     }

     free(used_nodes);

     return 0;
}

/* create congestor and network test subcomms letting the calling process no which they are */
int congestion_subcomms(CommConfig_t *config, CommNodes_t *nodes, int *congestor_node_list, 
                        int list_size, int *am_congestor, MPI_Comm *subcomm)
{
     int i, j, k, grank, group_size;
     MPI_Group base_group;
     MPI_Group ntwk_comm_group, congestor_comm_group;
     MPI_Comm null_comm;
     CommNode_t **used_nodes;
     int *congestors_group_list;

     mpi_error(MPI_Comm_group(MPI_COMM_WORLD, &base_group));

     /* determine the number of congestors */
     used_nodes = malloc(sizeof(CommNode_t *) * list_size);
     if (used_nodes == NULL) {
          die("Failed to allocate used_nodes in congestion_subcomms()\n");
     }
     group_size = 0;
     k = 0;
     for (i = 0; i < list_size; i++) {
          CommNode_t *tmp = nodes->nodes_head;
          for (j = 0; j < nodes->nnodes; j++) {
               if (tmp->node_id == congestor_node_list[i]) {
                    group_size += tmp->ppn;
                    used_nodes[k] = tmp;
                    k++;
               }
               tmp = tmp->next;
          }
     }

     congestors_group_list = malloc(sizeof(int) * group_size);
     if (congestors_group_list == NULL) {
          die("Failed to allocate congestors_group_list in congestion_subcomms()\n");
     }

     /* add in the ranks in the list */
     grank = 0;
     *am_congestor = 0;
     for (i = 0; i < list_size; i++) {
          CommRank_t *rtmp = used_nodes[i]->ranks_head;
          for (j = 0; j < used_nodes[i]->ppn; j++) {
               congestors_group_list[grank] = rtmp->rank;
               if (config->myrank == rtmp->rank) *am_congestor = 1;
               rtmp = rtmp->next;
               grank++;
          }
     }

     /* create the congestor and network test groups */
     mpi_error(MPI_Group_incl(base_group, group_size, congestors_group_list,
                              &congestor_comm_group));
     mpi_error(MPI_Group_excl(base_group, group_size, congestors_group_list,
                              &ntwk_comm_group));

     /* create our subcomm */
     if (*am_congestor) {
          mpi_error(MPI_Comm_create(MPI_COMM_WORLD, congestor_comm_group, subcomm));
     } else{
          mpi_error(MPI_Comm_create(MPI_COMM_WORLD, MPI_GROUP_EMPTY, &null_comm));
     }
     if (*am_congestor) {
          mpi_error(MPI_Comm_create(MPI_COMM_WORLD, MPI_GROUP_EMPTY, &null_comm));
     } else {
          mpi_error(MPI_Comm_create(MPI_COMM_WORLD, ntwk_comm_group, subcomm));
     }

     free(congestors_group_list);
     free(used_nodes);

     return 0;
}
