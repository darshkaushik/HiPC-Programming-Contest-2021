// G5
// Finding number of K-Cliques in an undirected graph
// Find made iterative, one thread per subtree

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


__global__ void find_iterative(int *d_k, bool *G_linear, int *imp, int *d_imp_size, int *cnt)
{
    int rootIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int imp_size = (*d_imp_size);
 
    if (rootIdx>=imp_size)return;
 
    int k = (*d_k);
    int root = imp[rootIdx];
    int thread_count=0;
    int lvl = 2;
    bool** lvl_vertices = (bool**)malloc((k + 1) * sizeof(bool*));
    int* num_lvl_vertices = (int*)malloc((k + 1) * sizeof(int));
    int* cur_vertex_id = (int*)malloc((k + 1) * sizeof(int));

    // The part of G_linear from root*imp_size to root*imp_size + imp_size - 1;
    lvl_vertices[lvl] = &(G_linear[rootIdx * imp_size]);
    
    // printf("inside Kernel, root = %d \n", root);
    // for(int i = 0; i < imp_size; i++)
    //     printf("%d ", lvl_vertices[lvl][i]);
    // printf("\n");
   
    cur_vertex_id[lvl] = 0;
    while(cur_vertex_id[lvl] < imp_size)
    {
        // printf("root = %d, lvl = %d, cur_vertex_id = %d \n", root, lvl, cur_vertex_id[lvl]);
        
        if(lvl_vertices[lvl][cur_vertex_id[lvl]] == 0)
        {
            // printf("Continued from cur_vertex_id = %d \n", cur_vertex_id[lvl]);
            cur_vertex_id[lvl]++;
            while(cur_vertex_id[lvl] == imp_size && lvl > 2)
            {
                // Go to parent level
                lvl = lvl - 1;

                // Go to parent's next sibling
                cur_vertex_id[lvl]++;
            }
            continue;
        }
        
        // vertex = imp[cur_vertex_id[lvl]];
        // vertex's adjacency list is the part of G_linear from vertex_id * imp_size to vertex_id * imp_size + imp_size - 1
        bool* adj_vertex = (bool*)malloc(imp_size * sizeof(bool));
        adj_vertex = &(G_linear[cur_vertex_id[lvl] * imp_size]);
            
        // printf("Till now ->\t\t\t");
        // for(int i = 0; i < imp_size; i++)
        //     printf("%d ", lvl_vertices[lvl][i]);
        // printf("\n");
        // printf("Adjacency list ->\t");
        // for(int i = 0; i < imp_size; i++)
        //     printf("%d ", adj_vertex[i]);
        // printf("\n");
            
        // intersec of adj_vertex[] with lvl_vertices[lvl][]
        lvl_vertices[lvl + 1] = (bool*)malloc(imp_size * sizeof(bool));
        num_lvl_vertices[lvl + 1] = 0;
        for(int i = 0; i < imp_size; i++)
        {
            lvl_vertices[lvl + 1][i] = (lvl_vertices[lvl][i] & adj_vertex[i]);
            if(lvl_vertices[lvl + 1][i] == 1)
            {
                num_lvl_vertices[lvl + 1]++; 
            }
        }
        
        // printf("Intersection ->\t\t");
        // for(int i = 0; i < imp_size; i++)
        //     printf("%d ", lvl_vertices[lvl + 1][i]);
        // printf("\n");
        // printf("num_lvl_vertices = %d \n \n", num_lvl_vertices[lvl + 1]);
        
        if(num_lvl_vertices[lvl + 1] > 0 && lvl + 1 < k)
        {
            lvl++;
            cur_vertex_id[lvl] = 0;
        }
        else
        {
            if(lvl + 1 == k)
            {
                thread_count+=num_lvl_vertices[lvl + 1]; 
            }

            // Go to next sibling
            cur_vertex_id[lvl]++;

            while(cur_vertex_id[lvl] == imp_size && lvl > 2)
            {
                // Go to parent level
                lvl = lvl - 1;

                // Go to parent's next sibling
                cur_vertex_id[lvl]++;
            }
        }
    }
    
    cnt[rootIdx] = thread_count;
    //printf("\nTotal count = %d\n--------------------------------------\n\n", thread_count);
}

int main()
{
    #ifndef ONLINE_JUDGE
    freopen("./input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
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

    // cout<<"printing degrees\n";
    // for(int i=0;i<n;i++)
    // {
    //     cout<<i<<" "<<d[i]<<"\n";
    // }
    // cout<<"\n";

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
    
    cout<<"Printing important vec of size = "<< imp_size<<endl; 
    // for(int i=0;i<imp_size;i++)
    // {
    //     cout<<imp[i]<<" ";
    // }
    // cout<<"\n\n";

    /*
    need imp, imp_size, k, i, cnt, v, v_size in gpu memory 
    d_v_size and d_v are already in gpu memory
    remaining imp, imp_size, k, i and cnt
    */

    int *cnt = (int*)malloc(imp_size*sizeof(int));
    int *d_imp, *d_imp_size, *d_k, *d_cnt;
    cudaMalloc(&d_imp, imp_size*sizeof(int));
    cudaMalloc(&d_imp_size, sizeof(int));
    cudaMalloc(&d_k, sizeof(int));
    cudaMalloc(&d_cnt, imp_size*sizeof(int));
    cudaCheckErrors("cudaMalloc failure");
    cudaMemset(d_cnt, 0, imp_size*sizeof(int));
    cudaCheckErrors("cudaMemset failure");
    cudaMemcpy(d_imp, imp, imp_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_imp_size, &imp_size, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, &k, sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy failure");
    
    // making G linear
    bool *G_linear = (bool*)malloc(imp_size * imp_size * sizeof(bool));
    for(int i = 0; i < imp_size; i++)
    {
        int vertex1 = imp[i];
        for(int j = 0; j < imp_size; j++)
        {
            int vertex2 = imp[j];
            G_linear[i*imp_size + j] = (mp.find({vertex1, vertex2}) != mp.end());
        }
    }
    
    // printf("Printing G_linear \n");
    // for(int i = 0; i < imp_size * imp_size; i++)
    // {
    //     if(i%imp_size==0)cout<<"\n";
    //     printf("%d ", G_linear[i]);
    // }
    // printf("\n \n");

    // storing G_linear in gpu
    bool *d_G_linear;
    cudaMalloc(&d_G_linear, imp_size * imp_size * sizeof(bool));
    cudaCheckErrors("cudaMalloc G_linear failure");
    cudaMemcpy(d_G_linear, G_linear, imp_size * imp_size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy G_linear failure");

    
    cout<<"start "<<(imp_size+1023)/1024<<endl;
    find_iterative<<<(imp_size+1023)/1024,1024>>>(d_k, d_G_linear, d_imp, d_imp_size, d_cnt);

    // End Time
    auto end_time = high_resolution_clock::now();
//------------------------ ALGORITHM Ends ----------------------------> 

//------------------------ OUTPUT Starts -----------------------------> 

    cudaMemcpy(cnt, d_cnt, imp_size*sizeof(int), cudaMemcpyDeviceToHost);
    long long ans=0;
    for(int i=0;i<12;i++)
    {
        ans+=cnt[i];
        cout<<cnt[i]<<"\n";
    }
    

    // Calculating time duration.
    auto duration = duration_cast<microseconds> (end_time - start_time);
    float time_us = duration.count();
    float time_ms = (float) duration.count() / 1000;
    float time_s = (float) duration.count() / 1000000;
    
    printf("%ld \n", ans);
    printf("Time Taken -> \n");
    printf("%.3f seconds \n", time_s);
    printf("%.3f milliseconds \n", time_ms);
    printf("%.3f microseconds \n", time_us);
//------------------------- OUTPUT Ends ------------------------------> 

}
