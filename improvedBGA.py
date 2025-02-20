import numpy as np
import random
import time
import pandas as pd

def read_in_data(file_path):
    """Reads problem data from a file and returns the constraint matrix and column costs."""
    # print(f"\n loading dataset: {file_path}")

    with open(file_path, 'r') as f:
        rows, cols = map(int, f.readline().split())
        con_matrix = np.zeros((rows, cols), dtype=int)  # Constraint matrix
        column_cost = np.zeros(cols, dtype=int)  # Cost per crew schedule

        coverage = [[] for _ in range(cols)]
        for j in range(cols):
            line = list(map(int, f.readline().split()))
            column_cost[j] = line[0]  # Store column cost
            num_flights = line[1]
            flights = [f - 1 for f in line[2:2 + num_flights]]  # Convert to 0-based indexing
            con_matrix[flights, j] = 1  # Assign crew coverage
            coverage[j] = flights  # Store flight coverage

    # print(f"loaded {rows} rows and {cols} columns")
    return con_matrix, column_cost, coverage, rows, cols

def initialize_population(num_ind, num_bit, con_matrix):
    """Generates initial population with some feasibility constraints."""
    pop = np.zeros((num_ind, num_bit), dtype=int)
    for i in range(num_ind):
        indices = np.random.choice(num_bit, size=np.random.randint(1, num_bit // 2), replace=False)
        pop[i, indices] = 1
        while np.any(con_matrix @ pop[i] < 1):  # Ensure feasibility
            new_idx = np.random.randint(0, num_bit)
            pop[i, new_idx] = 1
    return pop


def calculate_fitness(pop, con_matrix, column_cost):
    """Calculates fitness with penalty for infeasibility."""
    fitness = np.dot(pop, column_cost)  # Cost function
    constraint_violation = np.sum(np.maximum(0, 1 - con_matrix @ pop.T), axis=0)  # Uncovered flights
    penalty_factor = 10000  # Large penalty for infeasibility
    fitness += penalty_factor * constraint_violation
    return fitness, constraint_violation


def stochastic_ranking_sort(fitness, g_sum):
    """Sorts population based on fitness and constraint violations."""
    return np.lexsort((g_sum, fitness))  # Prioritize fitness, break ties with violations

def truncation_selection(pop, con_matrix, column_cost, num_parents):
    """Selects top individuals based on fitness."""
    fitness, g_sum = calculate_fitness(pop, con_matrix, column_cost)
    sorted_idx = stochastic_ranking_sort(fitness, g_sum)
    return pop[sorted_idx[:num_parents]]

def one_point_crossover(parent1, parent2):
    """Performs one-point crossover."""
    q = random.randint(1, len(parent1) - 1)
    return np.concatenate((parent1[:q], parent2[q:])), np.concatenate((parent2[:q], parent1[q:]))

def mutate(offspring, mutation_prob, con_matrix):
    """Mutate by ensuring feasibility (adds missing flights instead of random flips)."""
    for i in range(len(offspring)):
        if random.random() < mutation_prob:
            uncovered_flights = np.where(con_matrix @ offspring < 1)[0]
            if len(uncovered_flights) > 0:
                add_col = np.random.choice(np.where(con_matrix[uncovered_flights] > 0)[1])
                offspring[add_col] = 1  # Ensure feasibility
    return offspring


def heuristic_improvement_operator(individual, con_matrix, column_cost):
    """Improves individual feasibility."""
    for i in range(len(individual)):
        if individual[i] == 1:
            individual[i] = 0
            if np.sum(np.maximum(0, 1 - con_matrix @ individual)) > 0:
                individual[i] = 1  # Revert if infeasible
    return individual

def is_feasible(solution, coverage, rows):
    """Checks feasibility of a solution."""
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight] += 1
    return all(count >= 1 for count in flight_coverage)

def print_solution(solution, costs, coverage, rows):
    """Prints the solution in a readable format."""
    selected_schedules = [(i, costs[i], coverage[i]) for i in range(len(solution)) if solution[i] == 1]

    print("\n==== Genetic Algorithm Solution ====")
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
                flight_coverage[flight] += 1
    
    print("====================================\n")
    print("Flight coverage: ", flight_coverage)

def binary_genetic_algorithm(file_path, num_ind=40, max_iter=100, crossover_prob=0.85, mutation_prob=None):
    """Runs the Binary Genetic Algorithm (BGA)."""
    start_time = time.time()
    con_matrix, column_cost, coverage, rows, cols = read_in_data(file_path)
    num_bit = len(column_cost)
    mutation_prob = mutation_prob or (1 / num_bit)
    pop = initialize_population(num_ind, num_bit, con_matrix)
    
    for t in range(max_iter):
        parents = truncation_selection(pop, con_matrix, column_cost, num_ind // 3)
        offspring = parents.copy()
        
        for j in range(0, len(parents) - 1, 2):
            if random.random() < crossover_prob:
                offspring[j], offspring[j + 1] = one_point_crossover(parents[j], parents[j + 1])
        
        offspring = np.array([mutate(ind, mutation_prob, con_matrix) for ind in offspring])
        offspring = np.array([heuristic_improvement_operator(ind, con_matrix, column_cost) for ind in offspring])
        
        pop = np.vstack((pop, offspring))
        fitness, g_sum = calculate_fitness(pop, con_matrix, column_cost)
        sorted_idx = stochastic_ranking_sort(fitness, g_sum)
        pop = pop[sorted_idx[:num_ind]]
    
    best_fitness = fitness[sorted_idx[0]]
    best_solution = pop[0]
    feasibility = is_feasible(best_solution, coverage, rows)
    print(f"\n✅ Completed in {time.time() - start_time:.2f} seconds.")
    print(f"🏅 Best fitness: {best_fitness}")
    print(f"✅ Feasibility: {'Yes' if feasibility else 'No'}")
    print_solution(best_solution, column_cost, coverage, rows)
    return best_solution, best_fitness

def evaluate_algorithm(file_path, num_trials=10, max_iter=100):
    """Evaluates BGA over multiple trials."""
    successful_runs = 0
    total_cost = []
    total_times = []
    
    for _ in range(num_trials):
        start_time = time.time()
        solution, cost = binary_genetic_algorithm(file_path, max_iter=max_iter)
        feasible = is_feasible(solution, read_in_data(file_path)[2], read_in_data(file_path)[3])
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

# Example run
binary_genetic_algorithm('datasets/sppnw43.txt')

def evaluate_algorithm(file_path, num_trials=30, max_iter=100):
    """Evaluates BGA over multiple trials and returns average results and standard deviation."""
    successful_runs = 0
    total_cost = []
    total_times = []
    
    for _ in range(num_trials):
        start_time = time.time()
        solution, cost = binary_genetic_algorithm(file_path, max_iter=max_iter)
        feasible = is_feasible(solution, read_in_data(file_path)[2], read_in_data(file_path)[3])
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

# Run evaluation for each benchmark problem
benchmark_files = ["datasets/sppnw41.txt", "datasets/sppnw42.txt", "datasets/sppnw43.txt"]
benchmark_results = {file: evaluate_algorithm(file) for file in benchmark_files}

df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
print(df_results)

