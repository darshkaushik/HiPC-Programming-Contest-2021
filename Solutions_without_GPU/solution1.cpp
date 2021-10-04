// Finding number of K-Cliques in an undirected graph

#include <iostream>
#include <vector>
#include <set>
#include <map>
using namespace std;

const int N = 1e6;

vector<set<int>> v; // It will store graph similar to adjacency list but instead of list, set has been used.
int n,m,k,cnt;

// It will recurse and find all possible K-Cliques and increment cnt if a K-Clique is found.
void find(int i, set<int> s)
{
    if(n-i+1<s.size()) return;
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

    // first line of input should contain number of edges m and size of clique k
    cin >> m >> k;

    n = 0;
    map<pair<int,int>,int> mp;
    for(int i=0; i<m; i++)
    {
        int x,y;
        cin>>x>>y;
        if(x > y) swap(x,y);
        if(x!=y) mp[{x,y}] = 1;
        n = max(n, max(x,y));
    }
    n++;
    m = mp.size();

    // cout << n << " " << m << endl;
    
    vector<int> d(n,0); // d[i] will tell degree of node i.
    v.resize(n);
    for(auto [p,c]: mp)
    {
        int x = p.first, y = p.second;
        if(v[x].find(y)==v[x].end() && v[y].find(x)==v[y].end())
        {
            d[x]++;
            d[y]++;
        }

        if(x<y) v[x].insert(y);
        else v[y].insert(x);
    }

    set<int> imp; // Only those nodes will form k-clique that have degree >= k-1.
    for(int i=0; i<=n; i++)
    {
        if(d[i]>=k-1)
            imp.insert(i);
    }
    
    cnt=0;
    find(1,imp);

    cout<<cnt<<endl;
}

