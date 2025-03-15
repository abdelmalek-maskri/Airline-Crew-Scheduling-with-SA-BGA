# Air Crew Scheduling Optimization

This repository contains implementations of various optimization algorithms for solving the **Air Crew Scheduling Problem**. The objective is to minimize the total cost of crew schedules while ensuring all flight legs are covered exactly once.

## 📌 Problem Overview
The air crew scheduling problem is a **Set Partitioning Problem (SPP)** where we need to assign crews to flights in a way that minimizes cost and meets coverage constraints. Given a set of flight legs and available crew schedules, the goal is to find the most cost-effective assignment.

## 🚀 Implemented Algorithms
This repository provides **three optimization approaches** for solving the problem:

### **1. Standard Binary Genetic Algorithm (StandardBGA)**  
📌 Located in: `standardBGA.py`
- Uses **tournament selection**, **one-point crossover**, and **mutation**.
- Fitness function penalizes infeasible solutions to improve convergence.
- Implements **elitism** to preserve the best solutions across generations.

### **2. Improved Binary Genetic Algorithm (ImprovedBGA)**  
📌 Located in: `improvedBGA.py`
- Adds **heuristic improvement operators** to refine solutions.
- Uses **stochastic ranking** to balance fitness and feasibility.
- Incorporates **uniform crossover** and **proportional bit-flip mutation**.

### **3. Simulated Annealing (SA)**  
📌 Located in: `simulatedAnnealing.py`
- Uses an **initial feasible solution** and generates **neighboring solutions**.
- Applies the **Metropolis criterion** for probabilistic acceptance of worse solutions.
- Implements **linear cooling** to gradually reduce temperature over iterations.

## 📂 Running the Code
1. **Install dependencies:**  
   ```bash
   pip install numpy pandas
   ```
2. **Prepare dataset:**  
   - Place flight schedule data in the `datasets/` directory.
   - Example datasets: `datasets/sppnw41.txt`, `datasets/sppnw42.txt`, `datasets/sppnw43.txt`

3. **Run an algorithm:**  
   Execute one of the following:
   ```bash
   python standardBGA.py
   ```
   ```bash
   python improvedBGA.py
   ```
   ```bash
   python simulatedAnnealing.py
   ```

4. **Check the results:**  
   - The console will display the selected schedules, total cost, and feasibility.
   - A summary of results is saved to `summary.csv`.

## 📊 Benchmarking and Evaluation
Each algorithm runs **30 trials** to collect statistical performance data:
- **Success Rate (%)**: The percentage of runs producing a feasible solution.
- **Average Cost**: The mean total cost of the selected schedules.
- **Standard Deviation**: The variation in cost across trials.
- **Average Execution Time (s)**: The time taken per run.

To run benchmarking:
```bash
python standardBGA.py --benchmark
python improvedBGA.py --benchmark
python simulatedAnnealing.py --benchmark
```

## 📖 Documentation
- **[Requirements](requirements.pdf)**: Dependencies and setup instructions.
- **[Manual](manual.pdf)**: Guide on using and modifying the code.
- **[Replication Guide](replication.pdf)**: Steps to reproduce results.

---
This project demonstrates **evolutionary and probabilistic optimization methods** for solving real-world scheduling challenges efficiently. 🚀

