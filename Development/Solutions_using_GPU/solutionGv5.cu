// Finding number of K-Cliques in an undirected graph
// Only parallelizing v5 for degree and v_size

#include <iostream>
#include <cstring>
#include <algorithm>
#include <map>
#include <chrono>

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
const int N = 1e8;
int* v[N];
int v_size[N];
int n,m,k,cnt;

// It will recurse and find all possible K-Cliques and increment cnt if a K-Clique is found.
void find(int i, int options[], int options_size)
{
    if(k-i+1 > options_size) return;
    if(i == k)
    {
        cnt += options_size;
        return;
    }

    for(int i1 = 0; i1 < options_size; i1++)
    {
        int x = options[i1];
        
        // Finding intersection of options and v[x]
        int intersec_size = 0;
        for(int i2 = 0; i2 < v_size[x]; i2++)
        {
            int nd = v[x][i2];
            if(binary_search(options, options + options_size, nd))
                intersec_size++;
        }
        
        int intersec[intersec_size];
        memset(intersec, 0, sizeof(intersec));
        for(int i2 = 0, j = 0; i2 < v_size[x]; i2++)
        {
            int nd = v[x][i2];
            if(binary_search(options, options + options_size, nd))
            {
                intersec[j] = nd;
                j++;
            }
        }

        // Recursion
        find(i+1, intersec, intersec_size);
    }
}

// It will store the degree of each node 
__global__ void degree(int *e1, int *e2, int *d, int *v_size, int m)
{
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    if (idx < m)
    {
        //printf("index = %d\n",idx);
        int x = e1[idx], y = e2[idx];
        // x is smaller than y
        //printf("index = %d\tx = %d\ty = %d\n", idx,x,y);
     
        int *dx = &d[x], *dy = &d[y];
        atomicAdd(dx,1);
        atomicAdd(dy,1);
        atomicAdd(&v_size[x],1);
    }
    
}

int main()
{
    #ifndef ONLINE_JUDGE
    freopen("./Test_Files/input001.txt", "r", stdin);
    freopen("output001Gv5.txt", "w", stdout);
    #endif

//--------------------------- INPUT Starts -----------------------------> 
    // First line of input should contain number of edges m and size of clique k.
    cin >> m >> k;

    n = 0;
    // map to remove duplicate edges
    map<pair<int,int>,int> mp; 
    for(int i=0; i<m; i++)
    {
        int x,y;
        cin >> x >> y;
        // x must smaller than y
        if(x > y) swap(x,y);
        if(x != y) mp[{x,y}] = 1;
        n = max(n, y);
    }
    n++;
    m = mp.size();

    // Storing unique edges in e1[i] - e2[i] 
    int e1[m], e2[m];
    int i = 0;
    for(auto x: mp)
    {
        e1[i] = x.first.first;
        e2[i] = x.first.second;
        i++;
    }
 
    //for(i=0;i<m;i++)
    //{
    //    cout<<e1[i]<<" "<<e2[i]<<endl;
    //}

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

    // Print this to know the number of nodes and unique edges.
    // cout << n << " " << m << endl;

    // d[i] will tell degree of node i.
    // int d[n], v_size[n];
    // memset(d, 0, sizeof(d));
    // memset(v_size, 0, sizeof(d));
    // for(auto it: mp)
    // {
    //     pair<int,int> p = it.first;
    //     int x = p.first, y = p.second;
    //     d[x]++;
    //     d[y]++;
    //     // x is smaller than y
    //     v_size[x]++;
    // }

    // degree of nodes in device
    int *d_d, *d_v_size;
    cudaMalloc(&d_d, n*sizeof(int));
    cudaMalloc(&d_v_size, n*sizeof(int));
    cudaCheckErrors("cudaMalloc degree failure");
    cudaMemset(d_d, 0, n*sizeof(int));
    cudaMemset(d_v_size, 0, n*sizeof(int));
    cudaCheckErrors("cudaMemset degree failure");

    int deg_block_sz = 256;
    //cout<< "kernel config "<<(m+deg_block_sz-1)/deg_block_sz << deg_block_sz<<endl;
    degree<<<(m+deg_block_sz-1)/deg_block_sz, deg_block_sz>>>(d_e1, d_e2, d_d, d_v_size, m);
    cudaCheckErrors("Kernel degree launch failure");
    // cudaDeviceSynchronize();

    int d[n];
    cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_size, d_v_size, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy degree failure");
    
    //for(int i = 0; i < n; i++)
    //    cout<<i<<" "<< d[i]<<endl;

    // Finding adjacency list v[] of graph
    for(int i = 0; i < n; i++)
        v[i] = (int*)malloc(v_size[i] * sizeof(int));

    int v_i[n];
    memset(v_i, 0, sizeof(v_i));
    for(auto it: mp)
    {
        pair<int,int> p = it.first;
        int x = p.first, y = p.second;
        // x is smaller than y
        v[x][v_i[x]] = y;
        v_i[x]++;
    }


    // Only those nodes will form k-clique that have degree >= k-1.
    int imp_size = 0;
    for(int i = 0; i < n; i++)
    {
        if(d[i] >= k - 1)
            imp_size++;
    }

    int imp[imp_size];
    memset(imp, 0, sizeof(imp)); 
    for(int i = 0, j = 0; i < n; i++)
    {
        if(d[i] >= k - 1)
        {
            imp[j] = i;
            j++;
        }
    }
    
    cnt=0;
    find(1, imp, imp_size);

    // End Time
    auto end_time = high_resolution_clock::now();
//------------------------ ALGORITHM Ends ----------------------------> 

//------------------------ OUTPUT Starts -----------------------------> 
    // Calculating time duration.
    auto duration = duration_cast<microseconds> (end_time - start_time);
    long double time_us = duration.count();
    long double time_ms = (long double) duration.count() / 1000;
    long double time_s = (long double) duration.count() / 1000000;

    cout << cnt << endl;
    cout << "Time Taken -> " << endl;
    cout << time_s << " seconds" << endl;
    cout << time_ms << " milliseconds" << endl;
    cout << time_us << " microseconds" << endl;
//------------------------- OUTPUT Ends ------------------------------> 

}
