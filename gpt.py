import random
import math
import time
import numpy as np
import pandas as pd

def parse_problem(file_path):
    """ Reads the problem file and extracts costs and flight coverage. """
    with open(file_path, 'r') as file:
        rows, cols = map(int, file.readline().split())

        costs = []
        coverage = []
        
        for _ in range(cols):
            parts = list(map(int, file.readline().split()))
            cost = parts[0]
            num_flights = parts[1]
            flights = set(parts[2:2 + num_flights])
            
            costs.append(cost)
            coverage.append(flights)
    
    return rows, cols, costs, coverage

def calculate_cost(solution, costs, coverage, rows, penalty_factor=100):
    """ Computes the total cost while applying penalties for uncovered flights. """
    total_cost = sum(costs[i] for i in range(len(solution)) if solution[i] == 1)
    
    # Flight coverage tracking
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1

    # Penalty for uncovered flights
    penalty = sum(abs(count - 1) for count in flight_coverage) * penalty_factor
    return total_cost + penalty

def is_feasible(solution, coverage, rows):
    """ Checks if all flights are covered at least once. """
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1

    return all(count >= 1 for count in flight_coverage)

def repair_solution(solution, costs, coverage, rows):
    """ Repairs solutions by adding missing flights and removing redundant schedules. """
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1

    # Step 1: Add missing rotations for uncovered flights
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

    # Step 2: Remove redundant rotations
    for i in range(len(solution)):
        if solution[i] == 1:
            can_remove = all(flight_coverage[flight - 1] > 1 for flight in coverage[i])
            if can_remove:
                solution[i] = 0
                for flight in coverage[i]:
                    flight_coverage[flight - 1] -= 1

    return solution

def advanced_neighborhood(solution, costs, coverage):
    """ Ensures better solution diversity by swapping instead of random flipping. """
    ones = [i for i in range(len(solution)) if solution[i] == 1]
    zeros = [i for i in range(len(solution)) if solution[i] == 0]

    if not zeros or not ones:
        return solution  # No possible swap, return as is

    remove_idx = random.choice(ones)  # Remove an existing schedule
    add_idx = random.choice(zeros)    # Add a new one

    solution[remove_idx] = 0
    solution[add_idx] = 1

    return solution

def simulated_annealing(rows, cols, costs, coverage, max_iter=10000, initial_temp=100, cooling_rate=0.99, min_temp=1e-3, no_improve_limit=1000):

    """ Optimized Simulated Annealing with early stopping and repair. """
    current_solution = [random.randint(0, 1) for _ in range(cols)]
    best_solution = current_solution[:]
    best_cost = calculate_cost(current_solution, costs, coverage, rows)

    temperature = initial_temp
    no_improve_count = 0  # Track how many iterations go without improvement

    for iteration in range(max_iter):
        # Early stopping if temperature is too low or no improvement for a long time
        if temperature < min_temp or no_improve_count > no_improve_limit:
            break

        # Apply advanced neighborhood function
        neighbor = current_solution[:]
        flip_index = random.randint(0, cols - 1)
        neighbor[flip_index] = 1 - neighbor[flip_index]

        neighbor_cost = calculate_cost(neighbor, costs, coverage, rows)

        # Normalized acceptance probability
        delta_cost = neighbor_cost - best_cost
        scaled_delta = delta_cost / (best_cost + 1e-6)  # Prevent division by zero
        acceptance_prob = math.exp(-scaled_delta / temperature)

        # Accept new solution based on SA rules
        if delta_cost < 0 or random.random() < acceptance_prob:
            current_solution = neighbor
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost
                no_improve_count = 0  # Reset no improvement counter
            else:
                no_improve_count += 1  # Increase counter when no improvement

        # Adaptive cooling schedule
        temperature *= cooling_rate if iteration < max_iter // 2 else 0.995

    # Apply repair only if the final solution is infeasible (runs once, not inside the loop)
    if not is_feasible(best_solution, coverage, rows):
        best_solution = repair_solution(best_solution, costs, coverage, rows)
        best_cost = calculate_cost(best_solution, costs, coverage, rows)

    return best_solution, best_cost



def print_solution(solution, costs, coverage):
    selected_schedules = [(i, costs[i], coverage[i]) for i in range(len(solution)) if solution[i] == 1]
    
    print("\n==== Simulated Annealing Solution ====")
    print(f"Total Selected Crew Schedules: {len(selected_schedules)}")
    
    if not selected_schedules:
        print("No schedules selected.")
        return
    
    total_cost = sum(cost for _, cost, _ in selected_schedules)
    print(f"Total Cost: {total_cost}")
    print("\nSelected Crew Schedules:")
    
    for idx, cost, flights in selected_schedules:
        print(f"- Schedule {idx}: Cost = {cost}, Covers Flights: {sorted(flights)}")

    print("====================================\n")


rows, cols, costs, coverage = parse_problem("sppnw43.txt")

best_solution, best_cost = simulated_annealing(rows, cols, costs, coverage)

print_solution(best_solution, costs, coverage)
feasible = is_feasible(best_solution, coverage, rows)
print(feasible)
print("\n")


# def evaluate_algorithm(rows, cols, costs, coverage, num_trials=30):
#     """ Runs SA multiple times and collects performance statistics. """
#     successful_runs = 0
#     total_costs = []
#     total_times = []

#     for _ in range(num_trials):
#         start_time = time.time()
#         solution, cost = simulated_annealing(rows, cols, costs, coverage)
#         feasible = is_feasible(solution, coverage, rows)
#         duration = time.time() - start_time

#         if feasible:
#             successful_runs += 1
#             total_costs.append(cost)

#         total_times.append(duration)

#     success_rate = (successful_runs / num_trials) * 100
#     avg_cost = np.mean(total_costs) if total_costs else float('inf')
#     std_dev_cost = np.std(total_costs) if total_costs else float('inf')
#     avg_time = np.mean(total_times)

#     return {
#         "Success Rate (%)": success_rate,
#         "Average Cost": avg_cost,
#         "Standard Deviation": std_dev_cost,
#         "Average Execution Time (s)": avg_time
#     }

# # Load benchmark problem
# rows, cols, costs, coverage = parse_problem("sppnw41.txt")

# # Run and print results
# best_solution, best_cost = simulated_annealing(rows, cols, costs, coverage)
# print("\n==== Optimized Simulated Annealing Solution ====")
# print(f"Total Cost: {best_cost}")
# print(f"Solution Feasibility: {is_feasible(best_solution, coverage, rows)}\n")

# # Evaluate algorithm performance
# results = evaluate_algorithm(rows, cols, costs, coverage)
# print("Algorithm Performance:")
# for key, value in results.items():
#     print(f"{key}: {value:.2f}")


#evaluation if algo
def evaluate_algorithm(rows, cols, costs, coverage, num_trials=30):

    successful_runs = 0
    total_costs = []
    total_times = []

    for _ in range(num_trials):
        start_time = time.time()

        solution, cost = simulated_annealing(rows, cols, costs, coverage)
        # solution, cost = standard_bga(rows, cols, costs, coverage)
        # solution, cost = improved_bga(rows, cols, costs, coverage)

        feasible = is_feasible(solution, coverage, rows)

        duration = time.time() - start_time

        if feasible:
            successful_runs += 1
            total_costs.append(cost)

        total_times.append(duration)

    # Compute statistics
    success_rate = (successful_runs / num_trials) * 100
    average_cost = np.mean(total_costs) if total_costs else float('inf')
    std_dev_cost = np.std(total_costs) if total_costs else float('inf')
    average_time = np.mean(total_times)

    return {
        "Success Rate (%)": success_rate,
        "Average Cost": average_cost,
        "Standard Deviation": std_dev_cost,
        "Average Execution Time (s)": average_time
    }

# Load and evaluate each benchmark problem
benchmark_files = ["sppnw41.txt","sppnw42.txt","sppnw43.txt"]
benchmark_results = {}

for file_path in benchmark_files:
    rows, cols, costs, coverage = parse_problem(file_path)
    benchmark_results[file_path] = evaluate_algorithm(rows, cols, costs, coverage)

# Display results for each benchmark problem
df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
print(df_results)
