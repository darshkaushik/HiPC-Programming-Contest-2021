// v7
// Finding number of K-Cliques in an undirected graph
// Used binary representation instead of array for options, rest same as v6
// Only run on input with number of nodes having degree >= k - 1 is less than 60

#include <iostream>
#include <map>
#include <chrono>
using namespace std;
using namespace std::chrono;

// It will store graph like adjacency list.
int **v;
int *v_size;
int imp_size;
int *imp, *pos;
int n, m, k, cnt;

// It will recurse and find all possible K-Cliques and increment cnt if a K-Clique is found.
void find(int i, long long int options_bin, int options_size)
{
    if(k - i + 1 > options_size) return;
    if(i == k)
    {
        cnt += options_size;
        return;
    }

    for(int i1 = 0; i1 < imp_size; i1++)
    {
        // If i-th bit is not set in options_bin
        if((options_bin & (1 << i1)) == 0) 
            continue;
        
        int x = imp[i1];

        // Finding intersection of options and v[x]
        long long int intersec_bin = 0;
        int intersec_size = 0;
        for(int j = 0; j < v_size[x]; j++)
        {
            int nd = v[x][j];
            // if nd is present in imp and pos[nd]-th bit of options_bin is set
            if(pos[nd] != -1 && (options_bin & (1 << pos[nd])) != 0)
            {
                intersec_bin += (1 << pos[nd]);
                intersec_size ++;
            }
        }

        // Recursion
        find(i + 1, intersec_bin, intersec_size);
    }
}


int main()
{
    #ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    #endif

//--------------------------- INPUT Starts -----------------------------> 
    // First line of input should contain number of edges m and size of clique k.
    scanf("%d %d", &m, &k);

    n = 0;
    // map to remove duplicate edges
    map<pair<int, int>, int> mp; 
    for(int i=0; i<m; i++)
    {
        int x, y;
        scanf("%d %d", &x, &y);
        // x must smaller than y
        if(x > y) swap(x,y);
        if(x != y) mp[{x,y}] = 1;
        n = max(n, y);
    }
    n++;
    m = mp.size();
    v = (int**) malloc(n * sizeof(int*));
    v_size = (int*) calloc(n, sizeof(int));

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
//--------------------------- INPUT Ends -------------------------------> 

//------------------------ ALGORITHM Starts ----------------------------> 
    // Start Time
    auto start_time = high_resolution_clock::now();

    // Print this to know the number of nodes and unique edges.
    // printf("%d %d", n, m);

    // d[i] will tell degree of node i.
    int *d = (int*) calloc(n, sizeof(int));
    for(auto it: mp)
    {
        pair<int,int> p = it.first;
        int x = p.first, y = p.second;
        d[x]++;
        d[y]++;
        // x is smaller than y
        v_size[x]++;
    }

    // Finding adjacency list v[] of graph
    for(int i = 0; i < n; i++)
        v[i] = (int*) malloc(v_size[i] * sizeof(int));

    int *v_i = (int*) calloc(n, sizeof(int));
    for(auto it: mp)
    {
        pair<int,int> p = it.first;
        int x = p.first, y = p.second;
        // x is smaller than y
        v[x][v_i[x]] = y;
        v_i[x]++;
    }

    // Only those nodes will form k-clique that have degree >= k-1.
    // imp array stores nodes with degree >= k-1
    // pos[i] tells pos of node i in imp, pos[i] = -1 if node i not present in imp 
    imp_size = 0;
    for(int i = 0; i < n; i++)
    {
        if(d[i] >= k - 1)
            imp_size++;
    }

    pos = (int*) malloc(n * sizeof(int));
    imp = (int*) calloc(imp_size, sizeof(int));
    long long int imp_bin = 0;
    for(int i = 0, j = 0; i < n; i++)
    {
        if(d[i] >= k - 1)
        {
            imp[j] = i;
            pos[i] = j;
            imp_bin += (1 << j);
            j++;
        }
        else pos[i] = -1;
    }

    cnt=0;
    find(1, imp_bin, imp_size);

    // End Time
    auto end_time = high_resolution_clock::now();
//------------------------ ALGORITHM Ends ----------------------------> 

//------------------------ OUTPUT Starts -----------------------------> 
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
