import random
import math
import time
import numpy as np
import pandas as pd
from itertools import product

def parse_problem(file_path):
    with open(file_path, 'r') as f:
        rows, cols = map(int, f.readline().split())
        costs, coverage = [], []
        for _ in range(cols):
            parts = list(map(int, f.readline().split()))
            cost = parts[0]
            costs.append(cost)
            num_flights = parts[1]
            flights = set(parts[2 : 2 + num_flights])
            coverage.append(flights)
        return rows, cols, costs, coverage

def calculate_cost(solution, costs, coverage, rows, penalty_factor):
    total_cost = sum(costs[i] for i in range(len(solution)) if solution[i] == 1)
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1
    # Penalize if a flight is not covered exactly once
    penalty = sum(abs(count - 1) for count in flight_coverage) * penalty_factor
    return total_cost + penalty

def greedy_initial_solution(rows, cols, costs, coverage):
    solution = [0] * cols
    covered = set()
    # Continue until all flights are covered
    while len(covered) < rows:
        best_cost_ratio = float('inf')
        best_index = -1
        for i in range(cols):
            if solution[i] == 0:
                new_flights = coverage[i] - covered
                if new_flights:
                    cost_ratio = costs[i] / len(new_flights)
                    if cost_ratio < best_cost_ratio:
                        best_cost_ratio = cost_ratio
                        best_index = i
        if best_index != -1:
            solution[best_index] = 1
            covered.update(coverage[best_index])
        else:
            break
    return solution

def simulated_annealing(rows, cols, costs, coverage, max_iter=100000, initial_temp=1000, cooling_rate=0.99, min_temp=1e-3, no_improve_limit=1000, penalty_factor=50000):
    # initial solution.
    current_solution = greedy_initial_solution(rows, cols, costs, coverage)
    # If greedy fails to cover all flights, fall back to random initialization.
    if not is_feasible(current_solution, coverage, rows):
        current_solution = [random.randint(0, 1) for _ in range(cols)]
        current_solution = repair_solution(current_solution, costs, coverage, rows)

    current_cost = calculate_cost(current_solution, costs, coverage, rows, penalty_factor)
    best_solution = current_solution[:]
    best_cost = current_cost

    temperature = initial_temp
    no_improvement_count = 0

    for iteration in range(max_iter):
        if temperature < min_temp or no_improvement_count > no_improve_limit:
            break

        # Generate a neighbor by flipping one random bit.
        neighbour = current_solution[:]
        flip_index = random.randint(0, cols - 1)
        neighbour[flip_index] = 1 - neighbour[flip_index]

        # Repair neighbor if it becomes infeasible.
        if not is_feasible(neighbour, coverage, rows):
            neighbour = repair_solution(neighbour, costs, coverage, rows)

        neighbour_cost = calculate_cost(neighbour, costs, coverage, rows, penalty_factor)

        # Compare neighbour to current solution 
        delta_cost = neighbour_cost - current_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost/temperature):
            current_solution = neighbour
            current_cost = neighbour_cost
            if current_cost < best_cost:
                best_solution = current_solution[:]
                best_cost = current_cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1

        temperature *= cooling_rate

    # Final repair step if necessary.
    if not is_feasible(best_solution, coverage, rows):
        best_solution = repair_solution(best_solution, costs, coverage, rows)
        best_cost = calculate_cost(best_solution, costs, coverage, rows, penalty_factor)
    return best_solution, best_cost

def is_feasible(solution, coverage, rows):
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1
    # Require that each flight is covered exactly once.
    return all(count == 1 for count in flight_coverage)

def repair_solution(solution, costs, coverage, rows):
    # Identify flight coverage.
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1

    # Add missing rotations to cover uncovered flights (one flight at a time).
    for flight in range(rows):
        if flight_coverage[flight] == 0:
            best_rotation = -1
            best_cost = float('inf')
            for i in range(len(solution)):
                if (flight + 1) in coverage[i] and costs[i] < best_cost:
                    best_rotation = i
                    best_cost = costs[i]
            if best_rotation != -1:
                solution[best_rotation] = 1
                for f in coverage[best_rotation]:
                    flight_coverage[f - 1] += 1
                # Optionally, break here to only add one fix per repair call.
                break

    # Remove redundant rotations (over-covering flights).
    for i in range(len(solution)):
        if solution[i] == 1:
            if all(flight_coverage[flight - 1] > 1 for flight in coverage[i]):
                solution[i] = 0
                for flight in coverage[i]:
                    flight_coverage[flight - 1] -= 1

    return solution

def print_solution(solution, costs, coverage, rows):
    selected_schedules = [(i, costs[i], coverage[i]) for i in range(len(solution)) if solution[i] == 1]
    print("\n==== Simulated Annealing Solution ====")
    print(f"Total Selected Crew Schedules: {len(selected_schedules)}")
    if not selected_schedules:
        print("No schedules selected")
        return
    total_cost = sum(cost for _, cost, _ in selected_schedules)
    print(f"Total Cost: {total_cost}")
    print("\nSelected Crew Schedules:")
    for idx, cost, flights in selected_schedules:
        print(f"- Schedule {idx}: Cost = {cost}, Covers Flights: {sorted(flights)}")
    
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1
    print("====================================\n")
    print("Flight coverage: ", flight_coverage)

# Example run on one dataset
rows, cols, costs, coverage = parse_problem("datasets/sppnw43.txt")
best_sa_solution, best_sa_cost = simulated_annealing(rows, cols, costs, coverage)
print_solution(best_sa_solution, costs, coverage, rows)
feasible = is_feasible(best_sa_solution, coverage, rows)
print("Feasible solution:", feasible)

def evaluate_algorithm(rows, cols, costs, coverage, num_trials=30, max_iter=100000,
                                        initial_temp=1000,
                                        cooling_rate=0.99,
                                        min_temp=1e-3,
                                        no_improve_limit=1000,
                                        penalty_factor=50000):
    successful_runs = 0
    total_cost = []
    total_times = []
    for _ in range(num_trials):
        start_time = time.time()
        solution, cost = simulated_annealing(rows, cols, costs, coverage, max_iter, initial_temp, cooling_rate, min_temp, no_improve_limit, penalty_factor)
        feasible = is_feasible(solution, coverage, rows)
        duration = time.time() - start_time
        if feasible:
            successful_runs += 1
            total_cost.append(cost)
        total_times.append(duration)
    success_rate = (successful_runs / num_trials) * 100
    average_cost = np.mean(total_cost) if total_cost else float('inf')
    std_dev_cost = np.std(total_cost) if total_cost else float('inf')
    average_time = np.mean(total_times)
    return {
        "Success Rate (%)": success_rate,
        "Average Cost": average_cost,
        "Standard Deviation": std_dev_cost,
        "Average Execution Time (s)": average_time
    }

# Main execution and testing on benchmark datasets
benchmark_files = ["datasets/sppnw41.txt", "datasets/sppnw42.txt", "datasets/sppnw43.txt"]
benchmark_results = {}
for file_path in benchmark_files:
    rows, cols, costs, coverage = parse_problem(file_path)
    benchmark_results[file_path] = evaluate_algorithm(rows, cols, costs, coverage)
df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
print(df_results)







def parameter_tuning(file_path, num_trials=10):
    # Load the problem instance
    rows, cols, costs, coverage = parse_problem(file_path)
    
    # Define parameter ranges for tuning
    initial_temp_range = [1000, 2000, 5000, 3500]
    cooling_rate_range = [0.95, 0.99, 0.999, 0.9]
    max_iter_range = [1000, 10000, 100000, 20000]
    min_temp_range = [1e-3, 1e-4, 1e-5, 1e-6]
    no_improve_limit_range = [500, 1000, 2000, 3000]
    penalty_factor_range = [1000, 10000, 100000, 50000]
    
    # Store the results in a list of dictionaries
    results = []
    
    # Iterate over all combinations of parameters using product
    from itertools import product
    for params in product(initial_temp_range, cooling_rate_range, max_iter_range, min_temp_range, no_improve_limit_range, penalty_factor_range):
        initial_temp, cooling_rate, max_iter, min_temp, no_improve_limit, penalty_factor = params
        
        print(f"Testing parameters: initial_temp={initial_temp}, cooling_rate={cooling_rate}, max_iter={max_iter}, min_temp={min_temp}, no_improve_limit={no_improve_limit}, penalty_factor={penalty_factor}")
        evaluation = evaluate_algorithm(rows, cols, costs, coverage,
                                        num_trials=num_trials,
                                        max_iter=max_iter,
                                        initial_temp=initial_temp,
                                        cooling_rate=cooling_rate,
                                        min_temp=min_temp,
                                        no_improve_limit=no_improve_limit,
                                        penalty_factor=penalty_factor)
        # Include the parameter settings in the result
        evaluation["Parameters"] = {
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "max_iter": max_iter,
            "min_temp": min_temp,
            "no_improve_limit": no_improve_limit,
            "penalty_factor": penalty_factor
        }
        results.append(evaluation)
    
    # Create a DataFrame to show the results
    df_results = pd.DataFrame(results)
    return df_results


file_path = "datasets/sppnw42.txt"  
tuning_results = parameter_tuning(file_path, num_trials=30)
print(tuning_results)



