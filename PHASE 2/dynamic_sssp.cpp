#include <mpi/mpi.h>
#include <metis.h>
#include <omp.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>
#include <cassert>
#include <string>

using namespace std;

const float INF = std::numeric_limits<float>::infinity();

//
// Graph structure stored in CSR format.
//
struct Graph {
    idx_t nvtxs;                  // number of vertices
    vector<idx_t> xadj;           // CSR pointer for each vertex (size: nvtxs+1)
    vector<idx_t> adjncy;         // concatenated list of adjacent vertices
    vector<float> weights;        // corresponding edge weights
};

//
// Parse input graph from a file (.graph or .mgraph format):
//   - Skip comment lines (lines starting with '%').
//   - The first non-comment line should contain: numVertices [numEdges] (we only use numVertices).
//   - Each subsequent non-comment line corresponds to one vertex (line 1 = vertex 0, etc).
//   - Tokens on a line are assumed to be neighbors. If the number of tokens is even, assume neighbor-weight pairs.
//   - We assume input neighbor indices are 1-indexed (so we subtract one).
//
Graph loadGraphFromFile(const string &filename) {
    Graph g;
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error: Cannot open file: " << filename << endl;
        exit(1);
    }
    
    string line;
    // Skip any comments or empty lines
    while(getline(infile, line)) {
        if(line.empty()) continue;
        if(line[0]=='%') continue;
        break;
    }
    
    // Read graph header: number of vertices and optionally the number of edges.
    stringstream header(line);
    int numVertices, numEdges;
    header >> numVertices;
    header >> numEdges; // numEdges is optional and not used here.
    g.nvtxs = numVertices;
    
    // Temporary storage for per-vertex neighbors and weights.
    vector< vector<idx_t> > tempAdj(numVertices);
    vector< vector<float> > tempWeights(numVertices);
    
    // Read each vertex's data.
    for (int i = 0; i < numVertices; i++) {
        if (!getline(infile, line)) break;
        if (line.empty() || line[0]=='%') {
            i--; // if we skip a comment or blank line, repeat for the same vertex index.
            continue;
        }
        stringstream ls(line);
        vector<string> tokens;
        string token;
        while(ls >> token) {
            tokens.push_back(token);
        }
        
        bool hasWeight = false;
        // If an even number of tokens, assume neighbor-weight pairs.
        if ((!tokens.empty()) && (tokens.size() % 2 == 0))
            hasWeight = true;
            
        if(hasWeight) {
            for (size_t j = 0; j < tokens.size(); j += 2) {
                int neighbor = stoi(tokens[j]);
                // Convert from 1-indexed to 0-indexed
                if(neighbor > 0) neighbor--;
                float w = stof(tokens[j + 1]);
                tempAdj[i].push_back(neighbor);
                tempWeights[i].push_back(w);
            }
        } else {
            for (size_t j = 0; j < tokens.size(); j++) {
                int neighbor = stoi(tokens[j]);
                if(neighbor > 0) neighbor--;
                tempAdj[i].push_back(neighbor);
                tempWeights[i].push_back(1.0f); // default weight
            }
        }
    }
    
    // Build CSR arrays.
    g.xadj.resize(g.nvtxs + 1, 0);
    for (int i = 0; i < g.nvtxs; i++) {
        g.xadj[i+1] = g.xadj[i] + tempAdj[i].size();
    }
    int totalEdges = g.xadj[g.nvtxs];
    g.adjncy.resize(totalEdges);
    g.weights.resize(totalEdges);
    
    int idx = 0;
    for (int i = 0; i < g.nvtxs; i++) {
        for (size_t j = 0; j < tempAdj[i].size(); j++) {
            g.adjncy[idx] = tempAdj[i][j];
            g.weights[idx] = tempWeights[i][j];
            idx++;
        }
    }
    
    infile.close();
    return g;
}

//
// Create a sample graph (used if no filename is provided).
//
Graph createSampleGraph() {
    Graph g;
    g.nvtxs = 6;
    // For vertex 0: two edges; vertex 1: two; vertex 2: one; vertex 3: one; vertex 4: one; vertex 5: zero.
    g.xadj = {0, 2, 4, 5, 6, 7, 7};
    g.adjncy = {1, 2,  2, 3,  3,  4,  5};
    g.weights = {1, 4,  2, 6,  3,  1,  2};
    return g;
}

//
// Partition the graph using METIS_PartGraphKway.
// The 'partitions' vector will contain the partition ID for each vertex.
//
void partitionGraph(const Graph &g, int numPartitions, vector<idx_t> &partitions) {
    partitions.resize(g.nvtxs);
    idx_t nvtxs = g.nvtxs;
    idx_t ncon = 1;
    idx_t objval;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    int ret = METIS_PartGraphKway(&nvtxs, &ncon,
                                  const_cast<idx_t*>(g.xadj.data()),
                                  const_cast<idx_t*>(g.adjncy.data()),
                                  NULL, NULL, NULL,
                                  &numPartitions, NULL, NULL,
                                  options, &objval, partitions.data());
    if (ret != METIS_OK) {
        cerr << "METIS_PartGraphKway failed." << endl;
        MPI_Abort(MPI_COMM_WORLD, ret);
    }
}

//
// Main function: Implements a hybrid MPI+OpenMP SSSP.
// It reads a graph file if a filename is provided; otherwise, it uses the sample graph.
//
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int myRank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    
    Graph g;
    vector<idx_t> partitions;
    
    // Rank 0 reads the graph.
    if (myRank == 0) {
        if (argc >= 2) {
            string filename = argv[1];
            g = loadGraphFromFile(filename);
            cout << "Graph loaded from file: " << filename << endl;
        } else {
            cout << "No filename provided. Using hardcoded sample graph." << endl;
            g = createSampleGraph();
        }
        partitionGraph(g, numProcs, partitions);
        cout << "Graph created and partitioned into " << numProcs << " parts." << endl;
    }
    
    // Broadcast number of vertices.
    idx_t nvtxs;
    if (myRank == 0)
        nvtxs = g.nvtxs;
    MPI_Bcast(&nvtxs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (myRank != 0) {
        g.nvtxs = nvtxs;
        g.xadj.resize(nvtxs + 1);
    }
    MPI_Bcast(g.xadj.data(), nvtxs + 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Broadcast the number of edges.
    idx_t nedges;
    if (myRank == 0)
        nedges = g.adjncy.size();
    MPI_Bcast(&nedges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(myRank != 0) {
        g.adjncy.resize(nedges);
        g.weights.resize(nedges);
    }
    MPI_Bcast(g.adjncy.data(), nedges, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g.weights.data(), nedges, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Broadcast METIS partition information.
    if(myRank != 0)
        partitions.resize(nvtxs);
    MPI_Bcast(partitions.data(), nvtxs, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Each process selects its local vertices (those with partitions[i] == myRank).
    vector<int> localVertices;
    for (idx_t i = 0; i < nvtxs; i++) {
        if (partitions[i] == myRank)
            localVertices.push_back(i);
    }
    
    cout << "Rank " << myRank << " has " << localVertices.size() << " local vertices." << endl;
    
    // Initialize global SSSP arrays.
    vector<float> dist(nvtxs, INF);
    vector<int> parent(nvtxs, -1);
    int source = 0; // Choose vertex 0 as the source.
    if (source < nvtxs) {
        dist[source] = 0;
        parent[source] = source;
    }
    
    // Distributed relaxation: iterative Bellman–Ford style update.
    bool globalChanged = true;
    int iteration = 0;
    while(globalChanged && iteration < 1000) { // Set an iteration limit to avoid infinite loops.
        bool localChanged = false;
        
        // Each process relaxes edges from its local vertices using OpenMP.
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < localVertices.size(); i++) {
            int v = localVertices[i];
            float d_v = dist[v];
            if (d_v == INF) continue;
            for (idx_t j = g.xadj[v]; j < g.xadj[v + 1]; j++) {
                int u = g.adjncy[j];
                float newDist = d_v + g.weights[j];
                // Critical update to make sure updates are thread-safe within the process.
                #pragma omp critical
                {
                    if(newDist < dist[u]) {
                        dist[u] = newDist;
                        parent[u] = v;
                        localChanged = true;
                    }
                }
            }
        }
        
        // Synchronize the "changed" flag across processes.
        int localChangeInt = localChanged ? 1 : 0;
        int globalChangeInt = 0;
        MPI_Allreduce(&localChangeInt, &globalChangeInt, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Merge global distance vectors using element-wise minimum.
        vector<float> globalDist(nvtxs, INF);
        MPI_Allreduce(dist.data(), globalDist.data(), nvtxs, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
        dist = globalDist;
        
        globalChanged = (globalChangeInt > 0);
        iteration++;
    }
    
        // Rank 0 prints the final SSSP distances.
        if(myRank == 0) {
            cout << "\nSSSP completed in " << iteration << " iterations." << endl;
            cout << "Final distances from source " << source << ":\n";
            for (idx_t i = 0; i < nvtxs; i++) {
                cout << "Vertex " << i << ": " << dist[i] 
                     << " (Parent: " << parent[i] << ")\n";
            }
        }
        
        // Rename gmon.out so each MPI rank has a unique profiling file.
        MPI_Barrier(MPI_COMM_WORLD);
        
        char fname[128];
        sprintf(fname, "gmon.out.%d", myRank);
        if (rename("gmon.out", fname) != 0) {
            perror("rename");
            cerr << "Rank " << myRank << ": Error renaming gmon.out to " << fname << endl;
        }
        
        MPI_Finalize();
        return 0;
    }