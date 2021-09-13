// Finding number of K-Cliques in an undirected graph

#include <bits/stdc++.h>
using namespace std;

const int N=1e5;

set<int> v[N]; // It will store graph similar to adjacency list but instead of list, set has been used. 
int n,k,cnt;

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
    cin>>n>>k;
    int m;
    cin>>m;

    vector<int> d(N,0); // d[i] will tell degree of node i.

    for(int i=0; i<m; i++)
    {
        int x,y; 
        cin>>x>>y;
        
        if(v[x].find(y)==v[x].end() && v[y].find(x)==v[y].end())
        {
            d[x]++; 
            d[y]++;
        }

        if(x<y) v[x].insert(y);
        else v[y].insert(x);
    }

    set<int> imp; // Only those nodes will form k-clique that have degree >= k-1.
    for(int i=1; i<=n; i++)
    {
        if(d[i]>=k-1) 
            imp.insert(i);
    }
    
    cnt=0;
    find(1,imp);

    cout<<cnt<<endl;
}