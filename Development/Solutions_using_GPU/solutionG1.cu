// Finding number of K-Cliques in an undirected graph
// Parallelized find function, but only 1 thread at a time <<<1,1>>>

#include <iostream>
#include <algorithm>
#include <map>
#include <chrono>
#include <assert.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

using namespace std;
using namespace std::chrono;

// It will store graph like adjacency list.
int *v;
int *v_size;
int n, m, k;

// It will recurse and find all possible K-Cliques and increment cnt if a K-Clique is found.
__global__ void find(int i, int *options, int options_size, int k, int *v, int *v_size, int *cnt)
{
    i++;
    if(k - i + 1 > options_size) return;
    if(i == k)
    {
        // (*cnt) += (*options_size);
        atomicAdd(cnt, options_size);
        return;
    }

    for(int i1 = 0; i1 < options_size; i1++)
    {
        int x = options[i1];
        
        // Finding intersection of options and v[x]
        int intersec_size = 0, vsz = v_size[x] - v_size[x-1];
        for(int i2 = 0; i2 < vsz; i2++)
        {
            int nd = v[v_size[x-1] + i2];
            for(int i3 = 0; i3 < options_size; i3++)
            {
                if(options[i3] == nd)
                    intersec_size++;
            }
        }

        int *intersec = (int*) malloc(intersec_size * sizeof(int));
        for(int i2 = 0, j = 0; i2 < vsz; i2++)
        {
            int nd = v[v_size[x-1] + i2];
            for(int i3 = 0; i3 < options_size; i3++)
            {
                if(options[i3] == nd)
                {
                    intersec[j] = nd;
                    j++;
                }
            }
        }
        
        // Launching kernel inside kernel
        find<<<1,1>>>(i, intersec, intersec_size, k, v, v_size, cnt);
    }
}

__global__ void solve(int *i, int *options, int *options_size, int *k, int *v, int *v_size, int *cnt)
{
    find<<<1,1>>>((*i), options, (*options_size), (*k), v, v_size, cnt);
}

// It will store the degree of each node 
__global__ void degree(int *e1, int *e2, int *d, int *v_size, int m)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < m)
    {
        int x = e1[idx], y = e2[idx];
        int *dx = &d[x], *dy = &d[y];
        atomicAdd(dx,1);
        atomicAdd(dy,1);
        atomicAdd(&v_size[x],1);
    }
    
}

__global__ void prefix_sum(int *v_size, int n)
{
    for(int i = 1; i < n; i++)
    {
        v_size[i] += v_size[i - 1];
    }
}

__global__ void adj(int *e1, int *e2, int *v, int *v_i, int *v_size, int m)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < m)
    {
        int x = e1[idx], y = e2[idx];
        // x is smaller than y
        int i = atomicAdd(&v_i[x], 1);
        i += v_size[x-1];
        assert(i<m);
        v[i]=y;
    }
    
}

int main()
{
    #ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    #endif

//--------------------------- INPUT Starts -----------------------------> 
    // First line of input should contain number of edges m and size of clique k.
    scanf("%d %d", &m, &k);

    n = 0;
    // map to remove duplicate edges
    map<pair<int,int>,int> mp; 
    for(int i=0; i<m; i++)
    {
        int x,y;
        scanf("%d %d", &x, &y);
        // x must smaller than y
        if(x > y) swap(x,y);
        if(x != y) mp[{x,y}] = 1;
        n = max(n, y);
    }
    n++;
    m = mp.size();

    // Storing unique edges in e1[i] - e2[i] 
    int *e1 = (int*) malloc(m * sizeof(int));
    int *e2 = (int*) malloc(m * sizeof(int));
    int i = 0;
    for(auto x: mp)
    {
        e1[i] = x.first.first;
        e2[i] = x.first.second;
        i++;
    }

    // edges in device
    int *d_e1, *d_e2;
    cudaMalloc(&d_e1, m*sizeof(int));
    cudaMalloc(&d_e2, m*sizeof(int));
    cudaCheckErrors("cudaMalloc edges failure");
    cudaMemcpy(d_e1, e1, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_e2, e2, m*sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy edges failure");

//--------------------------- INPUT Ends -------------------------------> 

//------------------------ ALGORITHM Starts ----------------------------> 
    // Start Time
    auto start_time = high_resolution_clock::now();

    // degree of nodes in device
    int *d_d, *d_v_size;
    cudaMalloc(&d_d, n*sizeof(int));
    cudaMalloc(&d_v_size, n*sizeof(int));
    cudaCheckErrors("cudaMalloc degree failure");
    cudaMemset(d_d, 0, n*sizeof(int));
    cudaMemset(d_v_size, 0, n*sizeof(int));
    cudaCheckErrors("cudaMemset degree failure");

    int deg_block_sz = 256;
    degree<<<(m+deg_block_sz-1)/deg_block_sz, deg_block_sz>>>(d_e1, d_e2, d_d, d_v_size, m);
    cudaCheckErrors("Kernel degree launch failure");
    prefix_sum<<<1,1>>>(d_v_size, n);
    cudaCheckErrors("Kernel prefix_sum launch failure");

    int d[n];
    v_size = (int*) malloc(n * sizeof(int));
    cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_size, d_v_size, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy degree failure");
    
    int *d_v, *d_v_i;
    cudaMalloc(&d_v, m*sizeof(int));
    cudaMalloc(&d_v_i, n*sizeof(int));
    cudaCheckErrors("cudaMalloc adjacency_matrix failure");
    cudaMemset(d_v_i, 0, n*sizeof(int));
    cudaCheckErrors("cudaMemset adjacency_matrix failure");

    adj<<<(m+deg_block_sz-1)/deg_block_sz, deg_block_sz>>>(d_e1, d_e2, d_v, d_v_i, d_v_size, m);
    cudaCheckErrors("Kernel adjacency_matrix launch failure");

    v = (int*) malloc(m * sizeof(int));
    cudaMemcpy(v, d_v, m*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy adjacency_matrix failure");


    /*
    I need imp, imp_size, k, i, cnt, v, v_size in gpu memory 
    d_v_size and d_v are already in gpu memory
    need to write code for imp, imp_size, k, i and cnt
    */

    // Only those nodes will form k-clique that have degree >= k-1.
    int imp_size = 0;
    for(int i = 0; i < n; i++)
    {
        if(d[i] >= k - 1)
            imp_size++;
    }
    int *imp = (int*) calloc(imp_size, sizeof(int));
    for(int i = 0, j = 0; i < n; i++)
    {
        if(d[i] >= k - 1)
        {
            imp[j] = i;
            j++;
        }
    }
    // cnt=0;
    // find(1, imp, imp_size);

    int cnt = 0;
    int *d_imp, *d_imp_size, *d_k, *d_i, *d_cnt;
    cudaMalloc(&d_imp, imp_size*sizeof(int));
    cudaMalloc(&d_imp_size, sizeof(int));
    cudaMalloc(&d_k, sizeof(int));
    cudaMalloc(&d_i, sizeof(int));
    cudaMalloc(&d_cnt, sizeof(int));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemset(d_i, 0, sizeof(int));
    cudaMemset(d_cnt, 0, sizeof(int));
    cudaCheckErrors("cudaMemset failure");
    cudaMemcpy(d_imp, imp, imp_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imp_size, &imp_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");

    solve<<<1,1>>>(d_i, d_imp, d_imp_size, d_k, d_v, d_v_size, d_cnt);

    // End Time
    auto end_time = high_resolution_clock::now();
//------------------------ ALGORITHM Ends ----------------------------> 

//------------------------ OUTPUT Starts -----------------------------> 

    cudaMemcpy(&cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculating time duration.
    auto duration = duration_cast<microseconds> (end_time - start_time);
    float time_us = duration.count();
    float time_ms = (float) duration.count() / 1000;
    float time_s = (float) duration.count() / 1000000;
    
    printf("%d \n", cnt);
    printf("Time Taken -> \n");
    printf("%.3f seconds \n", time_s);
    printf("%.3f milliseconds \n", time_ms);
    printf("%.3f microseconds \n", time_us);
//------------------------- OUTPUT Ends ------------------------------> 

}
