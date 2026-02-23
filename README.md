# Project 814 - HPC Optimization Study

This project focuses on genetic algorithm implementations and optimization on HPC (High Performance Computing) systems, with emphasis on CPU and GPU performance benchmarking.

## Project Overview

This is a research project on high-performance computing conducted at Kumoh National Institute of Technology (금오공과대학교). The project investigates the effectiveness of genetic algorithms for solving combinatorial optimization problems, with a focus on comparative performance analysis between CPU and GPU implementations. Key research areas include evolutionary algorithm optimization, parallel computing strategies, and HPC resource utilization for machine learning and search problems.

## Main Files

### Core Implementation Files
- **814_solution.py** - Primary implementation file containing the main genetic algorithm framework and core logic for problem-solving
- **814_cpu_count_first.py** - CPU-based optimization variant that prioritizes maximizing the number of generations/iterations within time constraints
- **814_cpu_score_first.py** - CPU-based optimization variant that prioritizes achieving the best fitness score while monitoring generation count
- **814_gpu_count_first.py** - GPU-accelerated implementation optimized for maximum iteration count using parallel computation
- **814_gpu_score_first.py** - GPU-accelerated implementation optimized for fitness score improvement using parallel evaluation and mutation

### Testing & Utilities
- **gpu_test.py** - GPU capability testing and performance benchmarking script to validate GPU availability and measure computational throughput
- **wrong_code_test.py** - Debugging and validation script used to test problematic code paths and verify algorithm correctness
- **algorithm_flow.txt** - Text documentation describing the algorithm workflow, data flow, and execution pipeline

### Output Files
- **best_output.txt** - Best results and statistics from standard single-run executions, including fitness scores and evolution metrics
- **best_output_double.txt** - Best results and statistics from extended/double-run executions for improved statistical confidence and performance metrics

## Jupyter Notebooks

### Genetic Algorithms & Research
- **deap_study.ipynb** - Comprehensive study notes and tutorials on the DEAP (Distributed Evolutionary Algorithms in Python) library, including examples and best practices
- **one_max_num_gen.ipynb** - Implementation and analysis of the One-Max test problem using genetic algorithms, demonstrating algorithm convergence and fitness progression

### Eight Puzzle Problem
- **eight_puzzle_implement.ipynb** - Basic implementation of an Eight Puzzle solver using genetic algorithms with standard mutation and crossover operators
- **eight_puzzle_custom_mate.ipynb** - Advanced Eight Puzzle solver featuring custom mutation operators and specialized genetic operators tailored for puzzle-specific problem characteristics

## Folders

### linux_backup/
**Backup and reference folder containing the entire project as deployed on HPC supercomputer server**

This folder preserves the Linux-optimized version of the project tested on the HPC cluster infrastructure:
- **814_solution.py** - Base solution implementation
- **814_solution_count_first.py** - Iteration count optimization variant
- **814_solution_score_first.py** - Fitness score optimization variant
- **814_solution_gpu_count.py** - GPU-accelerated count-first variant
- **814_solution_gpu_score.py** - GPU-accelerated score-first variant
- **814_solution_gpu_full.py** - Full GPU implementation variant
- **814_work.sh** - Main job submission script for HPC batch execution
- **814_work_lite.sh** - Lightweight version for quick testing
- **814_work_double.sh** - Extended execution script for dual-run experiments
- **jobscript_for_gpu.sh** - HPC-specific job submission script configured for GPU node allocation and CUDA environment setup
- **requirements.txt** - Python package dependencies list specifying versions optimized for the HPC Linux environment
- **helloworld.py** - Simple test script
- **JupyterProj/** - Extended collection of Jupyter notebooks developed and refined during HPC research sessions
- **qsub_log/** - Archive of job output and error logs from HPC queue submissions (PBS/TORQUE scheduler), useful for debugging and performance analysis

### JupyterProj/ (within linux_backup)
Comprehensive collection of Jupyter notebooks developed during HPC research, including:
- **814_solution.ipynb** - Interactive notebook version of main solution
- **814_solution.py** - Python script version in notebook directory
- **814_solution_count_only.ipynb** - Count-optimization-only variant for iteration throughput analysis
- **814_solution_score_only.ipynb** - Score-optimization-only variant for pure fitness maximization studies
- **814_work.sh** - Job submission script in notebook directory
- **eight_puzzle_implement.ipynb** - Basic puzzle solver
- **eight_puzzle_custom_mate.ipynb** - Custom mutation operators variant
- **eight_puzzle_custom_mutate.ipynb** - Alternative custom mutation implementation
- **eight_puzzle_simpleai.ipynb** - SimpleAI library-based solver
- **gpu_usage_test.ipynb** - GPU resource utilization and performance monitoring
- **one_max_gen_algorithm.ipynb** - One-Max problem with genetic algorithms
- **one_max_num_gen.ipynb** - Alternative One-Max implementation
- **deap_study.ipynb** - DEAP library study notes
- **inputTest.ipynb** - Input data processing notebook
- **input.txt** - Input data file
- **814_Proj/** - Additional project subdirectory

## Project Goals

1. Implement genetic algorithms for problem-solving
2. Optimize performance on both CPU and GPU
3. Benchmark and compare different optimization strategies
4. Deploy and test on HPC supercomputer infrastructure
5. Document results and performance metrics

## Key Concepts

- **DEAP Library**: Distributed Evolutionary Algorithms in Python
- **Eight Puzzle Problem**: Classic AI search problem
- **Genetic Algorithms**: Evolutionary computation approach
- **GPU Optimization**: Leveraging CUDA/GPU for parallel computation
- **HPC Computing**: High-performance cluster computing environment

## Usage Notes

### Project Structure
The project maintains two synchronized versions:
- **Windows/Local versions**: Main directory files for local development and initial testing on Windows machines
- **Linux/HPC versions**: `linux_backup/` folder containing the same code adapted and tested on the HPC supercomputer infrastructure

### Optimization Variants
The research explores multiple optimization strategies through different implementations:
- **CPU vs GPU Execution**: Comparing serial CPU processing against parallel GPU acceleration using CUDA
- **Optimization Metrics**: Count-first (maximizing iterations) vs Score-first (maximizing fitness) optimization targets
- **Execution Modes**: Single-run experiments for quick iteration vs double/extended runs for statistical robustness

### Running the Project
Use the main directory files for local testing and the `linux_backup/` files for HPC cluster deployment via the provided shell scripts.

---

Last Updated: HPC Research Project, Kumoh National Institute of Technology
