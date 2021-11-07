#include<bits/stdc++.h>
using namespace std;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// Generates numbers between l and r uniformly.
int rand(int l, int r)
{
    auto dist = uniform_int_distribution<int> (l, r);
    return dist(rng);
}

// print the input
void print( vector<pair<int, int>> edges, int n)
{
    int m = edges.size();
    int k = rand(3, 7);
    cout << m << " " << k << endl;
    for(auto x: edges) 
        cout << x.first << " " << x.second << endl;
}


// generates a n-clique graph
void k_clique(int n, int type)
{
    vector<pair<int, int>> edges;
    for(int i = 1; i <= n; i++)
    {
        for(int j = i+1; j <= n; j++)
        {
            if(type == 1)
            {
                edges.push_back({i, j});
            }
            else
            {
                if(rand(1,5) != 1)
                    edges.push_back({i, j});
            }
        }
    }
    print(edges, n);
}

// generate a random graph of n nodes
void random_graph(int n)
{
    vector<pair<int, int>> edges;
    int m = rand(n*(n-1)/3, n*(n-1)/2);
    for(int i = 1; i <= m; i++)
    {
        int x = rand(1,n), y = rand(1,n);
        edges.push_back({x,y});
    }
    print(edges, n);
}

int main()
{
    // Write the input file name in freopen.
    freopen("input.txt", "w", stdout);

    srand(time(0));

    // Enter the type of test case to generate.
    int type = 3;
    if(type == 1 || type == 2)
    {
        int n = rand(5,20);
        k_clique(n, type);
    }
    else if(type == 3)
    {
        int n = rand(5,100);
        random_graph(n);
    }
}
