// v1
// Finding number of K-Cliques in an undirected graph

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
void find(int i, set<int> s)
{
    if(k-i+1 > s.size()) return;
    if(i==k)
    {
        cnt+=s.size();
        return;
    }

    for(auto x:s)
    {
        set<int> s1;
        for(auto nd:v[x])
        {
            if(s.find(nd)!=s.end())
                s1.insert(nd);
        }
        find(i+1,s1);
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

    // d[i] will tell degree of node i.
    vector<int> d(n,0);
    v.resize(n);
    for(auto it: mp)
    {
        pair<int,int> p = it.first;
        int x = p.first, y = p.second;
        if(v[x].find(y)==v[x].end() && v[y].find(x)==v[y].end())
        {
            d[x]++;
            d[y]++;
        }

        if(x<y) v[x].insert(y);
        else v[y].insert(x);
    }

    // Only those nodes will form k-clique that have degree >= k-1.
    set<int> imp; 
    for(int i=0; i<=n; i++)
    {
        if(d[i]>=k-1)
            imp.insert(i);
    }
    
    cnt=0;
    find(1,imp);

    // End Time
    auto end_time = high_resolution_clock::now();

    // Calculating time duration
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
