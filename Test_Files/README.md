### Input Format

- The first line of input file contain two numbers - number of edges `m` and `k` of K-Clique.
- Next `m` lines contains two integers `x` and `y` which have an edge.
- There can be duplicate edges in the input.

1. input0xx - small input (nodes <= 30)
2. input10x - large input that take less time
3. input20x - large input that take high time 

### Output Format
- Output file contains the answer i.e. the number of K-Cliques in the graph.

## ANALYSIS

Small Input - Description
- 001 - 
- 002 - 
- 003 - 5 clique, k = 5
- 004 - 5 clique, k = 4
- 005 - two disconnected 5 cliques, k = 5
- 006 - two connected 5 cliques, k = 4
- 007 - 10 clique, k = 6
- 008 - incomplete 10 clique, k = 6
- 009 - incomplete 10 clique, k = 4
- 010 - created using generator, type = 3, k = 5
- 011 - same as 010 but k = 6
- 012 - same as 010 but k = 7


Large Input less time - Description - Time Taken
- 101 - taken from example datasets - v5 0.72 s, v3 0.95 s
- 102 - created using generator, type = 3, k = 4,  - v5 0.4 s, v3 0.73 s
- 103 - same as 102, k = 5  - v5 4.2 s, v3 8.4 s


Large Input high time - 
- 201 - example dataset wiki-Talk.txt - v5 175 s
- 202 - same as 102, k = 6 - v5 33 s, v3 95 s