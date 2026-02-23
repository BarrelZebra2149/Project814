# Project 814 - HPC Optimization Study

This project focuses on genetic algorithm implementations and optimization on HPC (High Performance Computing) systems, with emphasis on CPU and GPU performance benchmarking.

## Project Overview

This is a research project on high-performance computing conducted at Kumoh National Institute of Technology (금오공과대학교). The project involves implementing and optimizing genetic algorithms and solving problems like the Eight Puzzle using various computational approaches.

## Main Files

### Core Implementation Files
- **814_solution.py** - Main solution implementation with genetic algorithms
- **814_cpu_count_first.py** - CPU-optimized version prioritizing iteration count
- **814_cpu_score_first.py** - CPU-optimized version prioritizing fitness score
- **814_gpu_count_first.py** - GPU-optimized version prioritizing iteration count
- **814_gpu_score_first.py** - GPU-optimized version prioritizing fitness score

### Testing & Utilities
- **gpu_test.py** - GPU testing and benchmarking script
- **wrong_code_test.py** - Testing script for debugging
- **algorithm_flow.txt** - Algorithm flow documentation

### Output Files
- **best_output.txt** - Best results from standard execution
- **best_output_double.txt** - Best results from double/extended execution

## Jupyter Notebooks

### Genetic Algorithms & Research
- **deap_study.ipynb** - Study notes on DEAP (Distributed Evolutionary Algorithms in Python) library
- **one_max_num_gen.ipynb** - One-Max problem implementation with genetic algorithms

### Eight Puzzle Problem
- **eight_puzzle_implement.ipynb** - Basic Eight Puzzle solver implementation
- **eight_puzzle_custom_mate.ipynb** - Eight Puzzle with custom mutation operators

## Folders

### linux_leeunchong/
**Backup folder from HPC supercomputer server**

Contains the same project structure optimized and tested on a Linux HPC system:
- **814_solution*.py** - Multiple variants of the solution
- **814_work*.sh** - Bash scripts for job submission on HPC clusters
- **jobscript_for_gpu.sh** - HPC job submission script for GPU tasks
- **requirements.txt** - Python dependencies for the HPC environment
- **JupyterProj/** - Jupyter notebooks and related implementations run on HPC
- **qsub_log/** - Job submission logs from HPC queue system (PBS/TORQUE)

### JupyterProj/ (within linux_leeunchong)
Extended Jupyter notebook implementations including:
- Score-only and count-only optimization variants
- Eight Puzzle implementations with different strategies
- GPU usage testing notebooks
- Input/output testing notebooks

## Documentation

- **[HPC 활용 결과 보고서] 금오공과대학교 이은총.hwp** - Final HPC project report (Korean)
  - Contains results and analysis from the HPC runs

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

The project maintains two versions:
- **Windows/Local versions**: Main directory files for development
- **Linux/HPC versions**: `linux_leeunchong/` for supercomputer deployment

Different optimization strategies are tested with variations in:
- CPU vs GPU execution
- Fitness score prioritization vs iteration count prioritization
- Single vs double execution runs

---

Last Updated: HPC Research Project, Kumoh National Institute of Technology
