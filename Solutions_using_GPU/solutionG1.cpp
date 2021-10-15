// solution using GPU
// Finding number of K-Cliques in an undirected graph
// Solution without using set.

#include <iostream>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>
using namespace std;
using namespace std::chrono;

// It will store graph similar to adjacency list but instead of list, set has been used.
vector<vector<int>> v;
int n,m,k,cnt;

// It will recurse and find all possible K-Cliques and increment cnt if a K-Clique is found.
void find(int i, vector<int> options)
{
    if(k-i+1 > options.size()) return;
    if(i==k)
    {
        cnt += options.size();
        return;
    }

    for(auto x: options)
    {
        // Finding intersection of options and v[x]
        vector<int> intersec;
        for(auto nd: v[x])
        {
            if(binary_search(options.begin(), options.end(), nd))
                intersec.push_back(nd);
        }
        find(i+1,intersec);
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
    cin >> m >> k;

    vector<pair<int,int>> edges;
    for(int i = 0; i < m; i++)
    {
        int x, y;
        cin >> x >> y;
        edges.push_back({x, y});
    }
//--------------------------- INPUT Ends -------------------------------> 

//------------------------ ALGORITHM Starts ----------------------------> 
    // Start Time
    auto start_time = high_resolution_clock::now();

    n = 0;
    // map to remove duplicate edges
    map<pair<int,int>,int> mp; 
    for(int i = 0; i < m; i++)
    {
        int x = edges[i].first; 
        int y = edges[i].second;
        // x must be smaller than y
        if(x > y) swap(x, y);
        if(x != y) mp[{x, y}] = 1;
        n = max(n, y);
    }
    n++;
    m = mp.size();

    // Print this to know the number of nodes and unique edges.
    // cout << n << " " << m << endl;

    // d[i] will tell degree of node i.
    vector<int> d(n, 0);
    v.resize(n);
    for(auto it: mp)
    {
        pair<int,int> p = it.first;
        int x = p.first, y = p.second;
        d[x]++;
        d[y]++;
        // x is smaller than y
        v[x].push_back(y);
    }

    // Only those nodes will form k-clique that have degree >= k-1.
    vector<int> imp; 
    for(int i = 0; i < n; i++)
    {
        if(d[i] >= k - 1)
            imp.push_back(i);
    }
    
    cnt=0;
    find(1, imp);

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
