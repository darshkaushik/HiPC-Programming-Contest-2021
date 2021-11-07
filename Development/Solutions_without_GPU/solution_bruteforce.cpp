// Finding number of K-Cliques in an undirected graph
// Brute Force, don't run it for graph with for nodes > 20 as it will run very slow

#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <chrono>
using namespace std;
using namespace std::chrono;

// It will store graph similar to adjacency list but instead of list, set has been used.
vector<set<int>> v; 
int n,m,k,cnt;

// It will recurse and find all possible K-Cliques and increment cnt if a K-Clique is found.
void find(set<int> &s)
{
    if(s.size()==k)
    {
        int is_clique = 1;
        for(auto x1: s)
        {
            for(auto x2: s)
            {
                if(x1 == x2) continue;
                if(v[x1].find(x2) == v[x1].end())
                {
                    is_clique = 0;
                    break;
                }
            }
            if(!is_clique) break;
        }
        if(is_clique) cnt++;
        return;
    }

    int x;
    if(s.empty()) x = 0;
    else x = *s.rbegin();

    for(int nd = x+1; nd <=n; nd++)
    {
        s.insert(nd);
        find(s);
        s.erase(nd);
    }
}

int main()
{
    #ifndef ONLINE_JUDGE
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    #endif

    // First line of input should contain number of edges m and size of clique k.
    cin >> m >> k;

    n = 0;
    // map to remove duplicate edges
    map<pair<int,int>,int> mp; 
    for(int i=0; i<m; i++)
    {
        int x,y;
        cin >> x >> y;
        if(x > y) swap(x,y);
        if(x != y) mp[{x,y}] = 1;
        n = max(n, max(x,y));
    }
    n++;
    m = mp.size();

    // Print this to know no. of nodes and unique edges.
    // cout << n << " " << m << endl;

    // Start Time
    auto start_time = high_resolution_clock::now();

    v.resize(n);
    for(auto it: mp)
    {
        pair<int,int> p = it.first;
        int x = p.first, y = p.second;
        v[x].insert(y);
        v[y].insert(x);
    }
    
    cnt = 0;
    set<int> s;
    find(s);

    // End Time
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end_time - start_time);
    long double time_us = duration.count();
    long double time_ms = (long double) duration.count() / 1000;
    long double time_s = (long double) duration.count() / 1000000;

    cout << cnt << endl;
    cout << "Time Taken -> " << endl;
    cout << time_s << " seconds" << endl;
    cout << time_ms << " milliseconds" << endl;
    cout << time_us << " microseconds" << endl;  
}
