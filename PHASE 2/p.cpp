// p.cpp
// MPI + OpenMP + METIS implementation of parallel dynamic SSSP update

#include <mpi/mpi.h>
#include <omp.h>
#include <metis.h>
#include <chrono>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <iostream>
#include <fstream>

using idx_t = idx_t;
using real_t = float;
const real_t INF = std::numeric_limits<real_t>::infinity();

// Graph stored in CSR
struct Graph {
    idx_t nv;                // number of vertices
    idx_t ne;                // number of edges (undirected, so stored twice)
    std::vector<idx_t> xadj; // size nv+1
    std::vector<idx_t> adjncy; // size 2*ne
    std::vector<real_t> adjwgt; // size 2*ne
};

// Partition info from METIS
struct Partition {
    int rank;
    int nprocs;
    idx_t local_nv;
    std::vector<idx_t> vtxdist;  // size nprocs+1
    std::vector<idx_t> part;     // size nv, METIS partitioning
};

// SSSP state per vertex
struct VertexState {
    real_t dist;
    idx_t parent;
    bool affected;
    bool affected_del;
};

// Load graph from edge list (u v w per line)
Graph load_graph(const std::string &filename) {
    std::ifstream in(filename);
    idx_t u, v;
    real_t w;
    std::vector<std::tuple<idx_t, idx_t, real_t>> edges;
    idx_t maxv = 0;
    while (in >> u >> v >> w) {
        edges.emplace_back(u, v, w);
        edges.emplace_back(v, u, w);
        maxv = std::max(maxv, std::max(u, v));
    }
    Graph G;
    G.nv = maxv + 1;
    G.ne = edges.size() / 2;
    G.xadj.resize(G.nv + 1);
    G.adjncy.resize(edges.size());
    G.adjwgt.resize(edges.size());

    // Debug: Print number of vertices and edges
    std::cout << "Graph loaded. Number of vertices: " << G.nv << ", Number of edges: " << G.ne << "\n";

    // sort edges by source
    std::sort(edges.begin(), edges.end(), [](auto &a, auto &b){ return std::get<0>(a) < std::get<0>(b); });

    // build CSR
    idx_t epos = 0;
    G.xadj[0] = 0;
    for (idx_t i = 0; i < G.nv; i++) {
        while (epos < (idx_t)edges.size() && std::get<0>(edges[epos]) == i) {
            G.adjncy[epos] = std::get<1>(edges[epos]);
            G.adjwgt[epos] = std::get<2>(edges[epos]);
            epos++;
        }
        G.xadj[i+1] = epos;
    }
    return G;
}

// Partition graph via METIS
void partition_graph(const Graph &G, Partition &P) {
    idx_t nvtxs = G.nv;
    idx_t ncon = 1; // number of balancing constraints (usually 1)
    P.part.resize(nvtxs);
    idx_t objval;

    std::vector<idx_t> xadj_copy = G.xadj;
    std::vector<idx_t> adjncy_copy = G.adjncy;

    if (nvtxs == 0 || xadj_copy.empty() || adjncy_copy.empty()) {
        if (P.rank == 0) std::cerr << "Invalid graph structure passed to METIS\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::cout << "Partitioning graph: nvtxs=" << nvtxs << ", ncon=" << ncon << std::endl;
    std::cout << "xadj[0] = " << xadj_copy[0] << ", adjncy[0] = " << adjncy_copy[0] << std::endl;
    
    // Only rank 0 calls METIS
    if (P.rank == 0) {
        int ret = METIS_PartGraphKway(&nvtxs, &ncon,
                                      xadj_copy.data(), adjncy_copy.data(),
                                      NULL, NULL, NULL,
                                      &P.nprocs, NULL, NULL,
                                      NULL, &objval, P.part.data());

        if (ret != METIS_OK) {
            std::cerr << "METIS_PartGraphKway failed with error code " << ret << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast partitioning to all ranks
    MPI_Bcast(P.part.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);

    // Build vtxdist
    P.vtxdist.assign(P.nprocs + 1, 0);
    for (idx_t i = 0; i < nvtxs; i++) {
        P.vtxdist[P.part[i] + 1]++;
    }
    for (int i = 1; i <= P.nprocs; i++) {
        P.vtxdist[i] += P.vtxdist[i - 1];
    }
    P.local_nv = P.vtxdist[P.rank + 1] - P.vtxdist[P.rank];
}

// Determine if a vertex is local to this rank
inline bool is_local(idx_t v, const Partition &P) {
    return (v >= P.vtxdist[P.rank] && v < P.vtxdist[P.rank+1]);
}

// Map global vertex to local index
inline idx_t to_local(idx_t v, const Partition &P) {
    return v - P.vtxdist[P.rank];
}

// Step 1: process insertions/deletions
void process_changes(const Graph &G, std::vector<VertexState> &state,
                     const std::vector<std::tuple<idx_t, idx_t, real_t,bool>> &changes,
                     const Partition &P) {
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < changes.size(); i++) {
        auto [u,v,w,is_ins] = changes[i];
        bool local_u = is_local(u,P), local_v = is_local(v,P);
        // only process if one endpoint is local
        if (!local_u && !local_v) continue;
        idx_t lu = local_u ? to_local(u,P) : -1;
        idx_t lv = local_v ? to_local(v,P) : -1;
        if (!is_ins) {
            // deletion: if edge was in SSSP tree
            // we assume we track tree membership externally
            // find deeper endpoint
            real_t du = local_u ? state[lu].dist : INF;
            real_t dv = local_v ? state[lv].dist : INF;
            if (du != INF && dv != INF) {
                idx_t y_g = (du>dv? u: v);
                if (is_local(y_g,P)) {
                    idx_t yl = to_local(y_g,P);
                    state[yl].dist = INF;
                    state[yl].affected_del = true;
                    state[yl].affected = true;
                }
            }
        } else {
            // insertion: relax if improves
            real_t du = local_u ? state[lu].dist : INF;
            real_t dv = local_v ? state[lv].dist : INF;
            if (du + w < dv) {
                if (local_v) {
                    #pragma omp critical
                    {
                        if (state[lv].dist > du + w) {
                            state[lv].dist = du + w;
                            state[lv].parent = u;
                            state[lv].affected = true;
                        }
                    }
                }
            } else if (dv + w < du) {
                if (local_u) {
                    #pragma omp critical
                    {
                        if (state[lu].dist > dv + w) {
                            state[lu].dist = dv + w;
                            state[lu].parent = v;
                            state[lu].affected = true;
                        }
                    }
                }
            }
        }
    }
}

// Step 2a: propagate deletions
void propagate_deletions(const Graph &G, std::vector<VertexState> &state,
                         const Partition &P) {
    bool again = true;
    while (again) {
        again = false;
        #pragma omp parallel for schedule(dynamic)
        for (idx_t i = 0; i < P.local_nv; i++) {
            if (!state[i].affected_del) continue;
            state[i].affected_del = false;
            // for each child in the tree: not stored explicitly;
            // so scan all local vertices
            for (idx_t j = 0; j < P.local_nv; j++) {
                if (state[j].parent == (P.vtxdist[P.rank]+i)) {
                    state[j].dist = INF;
                    state[j].affected_del = true;
                    state[j].affected = true;
                    again = true;
                }
            }
        }
    }
}

void relax_affected(const Graph &G, std::vector<VertexState> &state, const Partition &P) {
    bool again = true;
    
    while (again) {
        again = false;

        #pragma omp parallel for schedule(dynamic)
        for (idx_t i = 0; i < P.local_nv; i++) {
            if (!state[i].affected) continue;
            state[i].affected = false;

            // Get the global vertex index for the local vertex
            idx_t v_g = P.vtxdist[P.rank] + i;
            real_t dv = state[i].dist;

            // Debug: Print the vertex and its distance before relaxation
            std::cout << "Relaxing vertex " << v_g << " with current distance: " << dv << "\n";

            // Relaxing neighbors
            for (idx_t e = G.xadj[v_g]; e < G.xadj[v_g + 1]; e++) {
                idx_t n_g = G.adjncy[e];
                real_t w = G.adjwgt[e];
                bool local_n = is_local(n_g, P);

                if (local_n) {
                    idx_t nl = to_local(n_g, P);
                    real_t dn = state[nl].dist;

                    // Relax the edge (v_g -> n_g)
                    if (dn > dv + w) {
                        // Debug: Print if a relaxation occurs
                        std::cout << "Relaxing edge (" << v_g << " -> " << n_g << ") with weight " << w << "\n";
                        state[nl].dist = dv + w;
                        state[nl].parent = v_g;
                        state[nl].affected = true;
                        again = true; // Mark that relaxation occurred
                    }

                    // Relax the edge (n_g -> v_g)
                    if (dv > dn + w) {
                        // Debug: Print if a relaxation occurs
                        std::cout << "Relaxing edge (" << n_g << " -> " << v_g << ") with weight " << w << "\n";
                        state[i].dist = dn + w;
                        state[i].parent = n_g;
                        state[i].affected = true;
                        again = true; // Mark that relaxation occurred
                    }
                }
            }
        }
    }
}



int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    Partition P;
    P.rank = rank;
    P.nprocs = nprocs;

    if (argc < 4) {
        if (!rank) std::cerr << "Usage: " << argv[0] << " graph.edgelist source changes.txt\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 1) load graph (all ranks)
    Graph G = load_graph(argv[1]);

    if (rank == 0) {
        std::cout << "Graph loaded. Number of vertices: " << G.nv << ", Number of edges: " << G.ne << "\n";
    }
    

    // 2) partition graph via METIS
    partition_graph(G, P);

    // 3) initialize SSSP state on each rank
    
    idx_t source = std::stoll(argv[2]);
    std::vector<VertexState> state(P.local_nv);
    // Debug: Print initial state of vertices
    if (rank == 0) {
        std::cout << "Initializing SSSP state for source vertex " << source << "\n";
    }
    for (idx_t i = 0; i < P.local_nv; i++) {
        idx_t v_g = P.vtxdist[rank] + i;
        if (rank == 0) {
            std::cout << "Vertex " << v_g << " initialized with distance: " 
                    << (v_g == source ? 0 : INF) << "\n";
        }
        state[i].dist = (v_g == source ? 0 : INF);
        state[i].parent = (v_g == source ? source : -1);
        state[i].affected = (v_g == source);
        state[i].affected_del = false;
}


    // 4) read changes from file
    std::vector<std::tuple<idx_t, idx_t, real_t, bool>> changes;
    // format: u v w ins(1)/del(0) per line
    std::ifstream cinp(argv[3]);
    idx_t u, v;
    real_t w;
    int ins;
    while (cinp >> u >> v >> w >> ins) {
        changes.emplace_back(u, v, w, ins == 1);
    }

    // 5) apply update
    process_changes(G, state, changes, P);
    propagate_deletions(G, state, P);
    relax_affected(G, state, P);

    // 6) gather final state from all ranks
    std::vector<VertexState> global_state;
    if (rank == 0) {
        global_state.resize(G.nv);
    }
    MPI_Gather(state.data(), P.local_nv * sizeof(VertexState), MPI_BYTE,
               global_state.data(), P.local_nv * sizeof(VertexState), MPI_BYTE,
               0, MPI_COMM_WORLD);

    // 7) Output results (only on rank 0)
    if (rank == 0) {
        std::cout << "Final SSSP results:\n";
        for (idx_t v = 0; v < G.nv; v++) {
            if (global_state[v].dist != INF) {
                std::cout << "Vertex " << v << ":\n";
                std::cout << "  Distance: " << global_state[v].dist << "\n";
                std::cout << "  Parent: " << global_state[v].parent << "\n";
            } else {
                std::cout << "Vertex " << v << " is unreachable.\n";
            }
        }
    }

    MPI_Finalize();
    return 0;
}
//compile_with:  mpicxx -fopenmp -o p p.cpp -lmetis
//run_with:  mpirun -np 4 ./p mdual.graph 0 changes.txt