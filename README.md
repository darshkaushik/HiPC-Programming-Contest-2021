# [HiPC Programming Contest 2021](https://hipc.org/programming/)
## Track B: Nvidia GPU Architecture
<hr>

## Compilation and Execution Instructions

Follow these steps to run the solution on a GPU device:

1. Clone or download this Repository. 

2. The final source codes are present in the `Final Source Codes` folder. 
It has three solutions: optimised CPU solution, GPU solution 1 and GPU solution 2.

3. To run the solutions, create a repository named `test` (it can be named something else as well). Change the working directory to `test`.

4. Create a `solution.cu` file and a `input.txt` file. Copy the solution to be executed in the `solution.cu` and the input in the `input.txt`. Make sure that the input format is correct.

5. Excute the `solution.cu` file using the command below. The output will be printed on the command-line screen.
```
$nvcc solution.cu 
$./a.out
```
If this doesn't work use:
```
$nvcc -arch=sm_35 -rdc=true solution.cu
$./a.out
```
Here `arch=sm_35` denotes the compute capabilty of the gpu. For compute capability of 6.0 use `sm_60`, similarly for 8.5 use `sm_85`. 

<hr>

## Input Format

The first line of input should contain two integers - the first one for the number of edges in the graph and the second one for the value of k for which the number of k-cliques needs to be found.

Next lines contains two integers each which represents an edge.

<hr>

## Output Format
The number of k-cliques found along with the execution time is printed.  

<hr>