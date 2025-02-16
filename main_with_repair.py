import random
import math
import time
import numpy as np
import pandas as pd

def parse_problem(file_path):
    with open(file_path, 'r') as file:
        rows, cols = map(int, file.readline().split())
        # print(f'rows: {rows}')
        # print(f'columns: {cols}')

        costs = []
        coverage = []
        
        for _ in range(cols):
            parts = list(map(int, file.readline().split()))
            cost = parts[0]  
            num_flights = parts[1]
            flights = set(parts[2:2 + num_flights])  
            
            costs.append(cost)
            coverage.append(flights)
        
        # print(f'rows: {rows}')
        # print(f'columns: {cols}')
        # print(f'costs: {costs}')
        # print(f'coverage: {coverage}')
    
    return rows, cols, costs, coverage

# parse_problem("sppnw41.txt")


def calculate_cost(solution, costs, coverage, rows):
    
    total_cost = sum(costs[i] for i in range(len(solution)) if solution[i] == 1)
    
    # Calculate penalty for uncovered or over-covered flights
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1
    
    penalty = sum(abs(count - 1) for count in flight_coverage) * 100  # Penalty factor
    return total_cost + penalty

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


def repair_solution(solution, costs, coverage, rows):

    # Step 1: Identify uncovered flights
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1  # Flights are 1-indexed
    
    # Step 2: Add missing rotations to cover uncovered flights
    for flight in range(rows):
        if flight_coverage[flight] == 0:
            # Find the cheapest rotation that covers this flight
            best_rotation = -1
            best_cost = float('inf')
            for i in range(len(solution)):
                if (flight + 1) in coverage[i] and costs[i] < best_cost:
                    best_rotation = i
                    best_cost = costs[i]
            
            # Add the best rotation to the solution
            if best_rotation != -1:
                solution[best_rotation] = 1
                # Update flight coverage
                for f in coverage[best_rotation]:
                    flight_coverage[f - 1] += 1
    
    # Step 3: Remove redundant rotations that over-cover flights
    for i in range(len(solution)):
        if solution[i] == 1:
            # Check if removing this rotation still leaves all flights covered
            can_remove = True
            for flight in coverage[i]:
                if flight_coverage[flight - 1] <= 1:
                    can_remove = False
                    break
            
            # If the rotation is redundant, remove it
            if can_remove:
                solution[i] = 0
                # Update flight coverage
                for flight in coverage[i]:
                    flight_coverage[flight - 1] -= 1
    
    return solution


def simulated_annealing(rows, cols, costs, coverage, max_iter=10000, initial_temp=100, cooling_rate=0.95):

    # Initialize a random feasible solution
    current_solution = [random.randint(0, 1) for _ in range(cols)]
    best_solution = current_solution[:]
    best_cost = calculate_cost(current_solution, costs, coverage, rows)

    temperature = initial_temp

    for iteration in range(max_iter):
        # Generate a neighboring solution by flipping a random bit
        neighbor = current_solution[:]
        flip_index = random.randint(0, cols - 1)
        neighbor[flip_index] = 1 - neighbor[flip_index]

        # Calculate the cost of the neighbor
        neighbor_cost = calculate_cost(neighbor, costs, coverage, rows)

        # Acceptance probability
        delta_cost = neighbor_cost - best_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
            current_solution = neighbor
            if neighbor_cost < best_cost:
                best_solution = neighbor
                best_cost = neighbor_cost

        # Cool down the temperature
        temperature *= cooling_rate

    # best_solution = repair_solution(best_solution, costs, coverage, rows)
    # best_cost = calculate_cost(best_solution, costs, coverage, rows)

    return best_solution, best_cost



def is_feasible(solution, coverage, rows):
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1  # Flights are 1-indexed
    
    print("Flight coverage:", flight_coverage)

    # Check if all flights are covered exactly once
    return all(count >= 1 for count in flight_coverage)


rows, cols, costs, coverage = parse_problem("sppnw41.txt")

best_solution, best_cost = simulated_annealing(rows, cols, costs, coverage)

print_solution(best_solution, costs, coverage)
feasible = is_feasible(best_solution, coverage, rows)
print(feasible)
print("\n")


def evaluate_algorithm(rows, cols, costs, coverage, num_trials=30):

    successful_runs = 0
    total_costs = []
    total_times = []

    for _ in range(num_trials):
        start_time = time.time()

        solution, cost = simulated_annealing(rows, cols, costs, coverage)
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

# for file_path in benchmark_files:
#     rows, cols, costs, coverage = parse_problem(file_path)
#     benchmark_results[file_path] = evaluate_algorithm(rows, cols, costs, coverage)

# Display results for each benchmark problem
df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
print(df_results)























































# def genetic_algorithm(rows, cols, costs, coverage, population_size=50, generations=100, mutation_rate=0.01):
#     """
#     Standard Binary Genetic Algorithm for the Set Partitioning Problem.
    
#     Args:
#         rows (int): Number of flights.
#         cols (int): Number of crew schedules.
#         costs (list): List of costs for each crew schedule.
#         coverage (list): List of sets, where each set contains the flights covered by a crew schedule.
#         population_size (int): Size of the population.
#         generations (int): Number of generations.
#         mutation_rate (float): Probability of mutation.
    
#     Returns:
#         best_solution (list): Best solution found (binary vector).
#         best_cost (float): Cost of the best solution.
#     """
#     # Initialize population
#     population = [[random.randint(0, 1) for _ in range(cols)] for _ in range(population_size)]

#     for generation in range(generations):
#         # Evaluate fitness of each individual
#         fitness = [1 / (calculate_cost(individual, costs, coverage, rows) + 1e-6) for individual in population]

#         # Select parents using tournament selection
#         parents = []
#         for _ in range(population_size):
#             tournament = random.sample(range(population_size), 2)
#             parents.append(population[tournament[0]] if fitness[tournament[0]] > fitness[tournament[1]] else population[tournament[1]])

#         # Crossover (one-point crossover)
#         offspring = []
#         for i in range(0, population_size, 2):
#             parent1, parent2 = parents[i], parents[i + 1]
#             crossover_point = random.randint(1, cols - 1)
#             child1 = parent1[:crossover_point] + parent2[crossover_point:]
#             child2 = parent2[:crossover_point] + parent1[crossover_point:]
#             offspring.extend([child1, child2])

#         # Mutation (bit-flip mutation)
#         for individual in offspring:
#             for i in range(cols):
#                 if random.random() < mutation_rate:
#                     individual[i] = 1 - individual[i]

#         # Replace population with offspring
#         population = offspring

#     # Find the best solution in the final population
#     best_solution = min(population, key=lambda x: calculate_cost(x, costs, coverage, rows))
#     best_cost = calculate_cost(best_solution, costs, coverage, rows)

#     return best_solution, best_cost