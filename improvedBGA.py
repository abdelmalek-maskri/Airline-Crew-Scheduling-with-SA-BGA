import numpy as np
import random
import time
import pandas as pd

def read_in_data(file_path):
    with open(file_path, 'r') as f:
        rows, cols = map(int, f.readline().split())

        costs = np.zeros(cols, dtype=int)  
        coverage = []  

        for schedule_id in range(cols):
            line = list(map(int, f.readline().split()))
            costs[schedule_id] = line[0]  
            num_covered_flights = line[1]
            covered_flights = set(line[2:2 + num_covered_flights])  
            coverage.append(covered_flights)  

    return costs, coverage, rows, cols

def initialize_population(population_size, chromosome_length, coverage, rows):
    population = np.zeros((population_size, chromosome_length), dtype=int)

    for individual in range(population_size):
        # Randomly select schedules until all flights are covered exactly once
        covered_flights = set()  # Track flights covered so far
        while len(covered_flights) < rows:
            available_schedules = [i for i in range(chromosome_length) if not covered_flights.intersection(coverage[i])]  # Schedules that don't overlap
            if not available_schedules:
                break  # No more non-overlapping schedules
            chosen_schedule = np.random.choice(available_schedules)
            population[individual, chosen_schedule] = 1
            covered_flights.update(coverage[chosen_schedule])  # Add flights to the set

    return population

def calculate_fitness(population, costs, coverage, rows):
    """Calculates fitness based on cost and calculates constraint violations separately."""
    # Calculate costs (objective function values)
    objective_values = np.dot(population, costs)
    
    # Calculate constraint violations
    constraint_violations = np.zeros(len(population), dtype=int)

    for i, individual in enumerate(population):
        flight_coverage_count = [0] * rows  # Track how many times each flight is covered
        for schedule_id in range(len(individual)):
            if individual[schedule_id] == 1:
                for flight in coverage[schedule_id]:
                    flight_coverage_count[flight - 1] += 1  # Increment coverage count

        # Count over-covered and uncovered flights
        constraint_violations[i] = sum(abs(count - 1) for count in flight_coverage_count)

    return objective_values, constraint_violations

def stochastic_ranking(population, objective_values, constraint_violations, p_f=0.45):
    """
    Implements stochastic ranking for constraint handling.
    
    Args:
        population: The current population
        objective_values: Array of objective function values (costs)
        constraint_violations: Array of constraint violation counts
        p_f: Probability of using only the objective function for comparison
             when both solutions are infeasible
    
    Returns:
        Indices of sorted population
    """
    n = len(population)
    indices = list(range(n))
    
    # Bubble sort with stochastic ranking
    for i in range(n):
        swapped = False
        for j in range(n-1):
            # Both solutions feasible - compare by objective function
            if constraint_violations[indices[j]] == 0 and constraint_violations[indices[j+1]] == 0:
                if objective_values[indices[j]] > objective_values[indices[j+1]]:
                    indices[j], indices[j+1] = indices[j+1], indices[j]
                    swapped = True
            # Both solutions infeasible - probabilistic comparison
            elif constraint_violations[indices[j]] > 0 and constraint_violations[indices[j+1]] > 0:
                if random.random() < p_f:
                    # Compare by objective function
                    if objective_values[indices[j]] > objective_values[indices[j+1]]:
                        indices[j], indices[j+1] = indices[j+1], indices[j]
                        swapped = True
                else:
                    # Compare by constraint violation
                    if constraint_violations[indices[j]] > constraint_violations[indices[j+1]]:
                        indices[j], indices[j+1] = indices[j+1], indices[j]
                        swapped = True
            # One solution feasible, one infeasible - feasible is better
            elif constraint_violations[indices[j]] > constraint_violations[indices[j+1]]:
                indices[j], indices[j+1] = indices[j+1], indices[j]
                swapped = True
                
        if not swapped:
            break
            
    return indices

def one_point_crossover(parent1, parent2):
    """Performs one-point crossover between two parent solutions."""
    crossover_point = random.randint(1, len(parent1) - 1)  
    return np.concatenate((parent1[:crossover_point], parent2[crossover_point:])), \
           np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

def mutate(offspring, mutation_probability, coverage, rows):
    """Applies mutation ensuring that each flight is covered exactly once."""
    for i in range(len(offspring)):
        if random.random() < mutation_probability:
            # Find flights covered by the current solution
            covered_flights = set()
            for schedule_id in range(len(offspring)):
                if offspring[schedule_id] == 1:
                    covered_flights.update(coverage[schedule_id])

            # Find uncovered flights
            uncovered_flights = set(range(1, rows + 1)) - covered_flights

            # Find schedules that cover uncovered flights without overlapping
            available_schedules = [schedule_id for schedule_id in range(len(offspring)) 
                                  if not covered_flights.intersection(coverage[schedule_id]) 
                                  and uncovered_flights.intersection(coverage[schedule_id])]

            if available_schedules:
                chosen_schedule = np.random.choice(available_schedules)
                offspring[chosen_schedule] = 1  # Add the schedule
                covered_flights.update(coverage[chosen_schedule])  # Update covered flights

    return offspring

def is_feasible(solution, flight_coverage_list, num_flights):
    """Checks whether all flights are covered exactly once."""
    flight_coverage_count = [0] * num_flights
    for schedule_id in range(len(solution)):
        if solution[schedule_id] == 1:
            for flight in flight_coverage_list[schedule_id]:
                flight_coverage_count[flight - 1] += 1
    return all(count == 1 for count in flight_coverage_count) 

def print_solution(solution, costs, coverage, rows):
    """Prints the selected schedules and total cost."""
    selected_schedules = [(schedule_id, costs[schedule_id], coverage[schedule_id]) 
                          for schedule_id in range(len(solution)) if solution[schedule_id] == 1]

    print("\n==== Genetic Algorithm Solution ====")
    print(f"Total Selected Crew Schedules: {len(selected_schedules)}")

    if not selected_schedules:
        print("No schedules selected")
        return
    
    total_cost = sum(cost for _, cost, _ in selected_schedules)
    print(f"Total Cost: {total_cost}")
    print("\nSelected Crew Schedules:")
    
    for schedule_id, cost, flights in selected_schedules:
        print(f"- Schedule {schedule_id}: Cost = {cost}, Covers Flights: {sorted(flights)}")
    
    flight_coverage_count = [0] * rows
    for schedule_id in range(len(solution)):
        if solution[schedule_id] == 1:
            for flight in coverage[schedule_id]:
                flight_coverage_count[flight - 1] += 1

    print("====================================\n")
    print("Flight coverage: ", flight_coverage_count)

def binary_genetic_algorithm_with_stochastic_ranking(file_path, population_size=100, max_generations=1000, 
                                                     crossover_probability=0.85, mutation_probability=None, p_f=0.45):
    """
    Binary Genetic Algorithm with Stochastic Ranking for the Set Covering Problem.
    
    Args:
        file_path: Path to the data file
        population_size: Size of the population
        max_generations: Maximum number of generations
        crossover_probability: Probability of crossover
        mutation_probability: Probability of mutation (defaults to 1/chromosome_length)
        p_f: Probability of comparing by objective function when both solutions are infeasible
    """
    costs, coverage, rows, cols = read_in_data(file_path)
    
    mutation_probability = mutation_probability or (1 / cols)
    population = initialize_population(population_size, cols, coverage, rows)
    
    for generation in range(max_generations):
        objective_values, constraint_violations = calculate_fitness(population, costs, coverage, rows)
        
        # Apply stochastic ranking
        sorted_indices = stochastic_ranking(population, objective_values, constraint_violations, p_f)
        population = population[sorted_indices[:population_size]]
        
        # Select parents for reproduction
        parents = population[:population_size // 3]
        offspring = parents.copy()

        # Apply crossover
        for j in range(0, len(parents) - 1, 2):
            if random.random() < crossover_probability:
                offspring[j], offspring[j + 1] = one_point_crossover(parents[j], parents[j + 1])
        
        # Apply mutation
        offspring = np.array([mutate(ind, mutation_probability, coverage, rows) for ind in offspring])
        
        # Update population with offspring
        population = np.vstack((population, offspring))

    # Get the best solution after all generations
    final_objective_values, final_constraint_violations = calculate_fitness(population, costs, coverage, rows)
    sorted_indices = stochastic_ranking(population, final_objective_values, final_constraint_violations, p_f)
    best_solution = population[sorted_indices[0]]
    best_fitness = np.dot(best_solution, costs)
    
    return best_solution, best_fitness

# Example usage
costs, coverage, rows, num_schedules = read_in_data('datasets/sppnw41.txt')
best_solution, best_fitness = binary_genetic_algorithm_with_stochastic_ranking('datasets/sppnw41.txt')
print_solution(best_solution, costs, coverage, rows)
feasible = is_feasible(best_solution, coverage, rows)
print(f"Solution is feasible: {feasible}")
print("\n")

def evaluate_algorithm(file_path, num_trials=30, max_generations=100):
    """Evaluates the genetic algorithm across multiple trials."""
    successful_runs = 0
    total_costs = []
    execution_times = []
    
    for trial in range(num_trials):
        print(f"Trial {trial+1}/{num_trials}...")
        start_time = time.time()
        solution, cost = binary_genetic_algorithm_with_stochastic_ranking(file_path, max_generations=max_generations)
        feasible = is_feasible(solution, read_in_data(file_path)[1], read_in_data(file_path)[2])
        duration = time.time() - start_time

        if feasible:
            successful_runs += 1
            total_costs.append(cost)
        execution_times.append(duration)

    success_rate = (successful_runs / num_trials) * 100
    average_cost = np.mean(total_costs) if total_costs else float('inf')
    std_dev_cost = np.std(total_costs) if total_costs else float('inf')
    average_time = np.mean(execution_times)

    return {
        "Success Rate (%)": success_rate,
        "Average Cost": average_cost,
        "Standard Deviation": std_dev_cost,
        "Average Execution Time (s)": average_time
    }

benchmark_files = ["datasets/sppnw41.txt", "datasets/sppnw42.txt", "datasets/sppnw43.txt"]
benchmark_results = {file: evaluate_algorithm(file) for file in benchmark_files}

df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
print(df_results)