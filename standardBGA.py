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

# costs, coverage, rows, cols = read_in_data("datasets/sppnw41.txt")
# print(f"rows {rows}")
# print(f"cols {cols}")
# print(f"coverage {coverage}")
# print(f"costs {costs}")

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
    """Calculates fitness based on cost and penalizes over-covered or uncovered flights."""
    fitness_values = np.dot(population, costs)  
    penalties = np.zeros(len(population), dtype=int)

    for i, individual in enumerate(population):
        flight_coverage_count = [0] * rows  # Track how many times each flight is covered
        for schedule_id in range(len(individual)):
            if individual[schedule_id] == 1:
                for flight in coverage[schedule_id]:
                    flight_coverage_count[flight - 1] += 1  # Increment coverage count

        # Penalize over-covered and uncovered flights
        penalties[i] = sum(abs(count - 1) for count in flight_coverage_count)  


    penalty = 10000 
    fitness_values += penalty * penalties  

    return fitness_values, penalties



def one_point_crossover(parent1, parent2):
    """Performs one-point crossover between two parent solutions."""
    crossover_point = random.randint(1, len(parent1) - 1)  
    return np.concatenate((parent1[:crossover_point], parent2[crossover_point:])), \
           np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

def two_point_crossover(parent1, parent2):
    """Performs two-point crossover between two parent solutions."""
    # Select two random crossover points
    crossover_point1 = random.randint(1, len(parent1) - 2)  # First point
    crossover_point2 = random.randint(crossover_point1 + 1, len(parent1) - 1)  # Second point

    # Create offspring by swapping the segment between the two points
    offspring1 = np.concatenate((
        parent1[:crossover_point1],  # Part before first point from parent1
        parent2[crossover_point1:crossover_point2],  # Segment between points from parent2
        parent1[crossover_point2:]  # Part after second point from parent1
    ))
    offspring2 = np.concatenate((
        parent2[:crossover_point1],  # Part before first point from parent2
        parent1[crossover_point1:crossover_point2],  # Segment between points from parent1
        parent2[crossover_point2:]  # Part after second point from parent2
    ))

    return offspring1, offspring2

def uniform_crossover(parent1, parent2):
    """Performs uniform crossover between two parent solutions."""
    # Create a mask to decide which parent contributes each gene
    mask = np.random.randint(2, size=len(parent1))  # Randomly choose 0 or 1 for each gene

    # Create offspring by selecting genes from parents based on the mask
    offspring1 = np.where(mask, parent1, parent2)  # If mask is 1, take from parent1; else, parent2
    offspring2 = np.where(mask, parent2, parent1)  # If mask is 1, take from parent2; else, parent1

    return offspring1, offspring2

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
    
def tournament_selection(population, fitness_values, penalties, tournament_size=2):
    """
    Tournament selection method that prioritizes feasibility and then cost.
    
    Args:
        population: The current population
        fitness_values: Fitness values for each individual (lower is better)
        penalties: Penalty values for constraint violations (lower is better)
        tournament_size: Number of individuals in each tournament
        
    Returns:
        Selected individuals for reproduction
    """
    population_size = len(population)
    selected = []
    
    # Select enough parents to maintain population size after crossover
    while len(selected) < population_size // 3:
        # Randomly select tournament_size individuals
        tournament_indices = random.sample(range(population_size), tournament_size)
        
        # Find the best individual in the tournament
        best_index = tournament_indices[0]
        for idx in tournament_indices[1:]:
            # First prioritize feasibility (lower penalties)
            if penalties[idx] < penalties[best_index]:
                best_index = idx
            # If both have same feasibility, prioritize cost (lower fitness)
            elif penalties[idx] == penalties[best_index] and fitness_values[idx] < fitness_values[best_index]:
                best_index = idx
        
        selected.append(population[best_index])
    
    return np.array(selected)

#can change the population size (decrease it for a better run time)
# Modify your binary_genetic_algorithm function to use tournament selection:
def binary_genetic_algorithm(file_path, population_size=100, max_generations=1000, 
                            crossover_probability=0.85, mutation_probability=None,
                            elite_percentage=0.1):  # Add elite percentage parameter
    
    costs, coverage, rows, cols = read_in_data(file_path)
    
    mutation_probability = mutation_probability or (1 / cols) 
    population = initialize_population(population_size, cols, coverage, rows)
    
    # Calculate number of elite individuals to preserve
    num_elite = max(1, int(population_size * elite_percentage))
    
    for generation in range(max_generations):
        fitness_values, penalties = calculate_fitness(population, costs, coverage, rows)
        
        # Sort population by penalties first, then cost
        sorted_indices = np.lexsort((fitness_values, penalties))  
        sorted_population = population[sorted_indices]
        
        # Store elite individuals
        elite_individuals = sorted_population[:num_elite].copy()
        
        # The rest remains similar
        sorted_fitness = fitness_values[sorted_indices[:population_size]]
        sorted_penalties = penalties[sorted_indices[:population_size]]
        
        # Use tournament selection for the remainder
        parents = tournament_selection(sorted_population[:population_size], 
                                      sorted_fitness, sorted_penalties)
        offspring = parents.copy()

        for j in range(0, len(parents) - 1, 2):
            if random.random() < crossover_probability:
                offspring[j], offspring[j + 1] = one_point_crossover(parents[j], parents[j + 1])
        
        offspring = np.array([mutate(ind, mutation_probability, coverage, rows) 
                             for ind in offspring])
        
        # Form new population including elite individuals
        population = np.vstack((elite_individuals, offspring))
        
        # If needed, add more random individuals to maintain population size
        if len(population) < population_size:
            additional_individuals = initialize_population(
                population_size - len(population), cols, coverage, rows)
            population = np.vstack((population, additional_individuals))
    
    # Return the best solution found
    best_solution = population[0]
    best_fitness = np.dot(best_solution, costs)
    
    return best_solution, best_fitness

# Example usage
costs, coverage, rows, num_schedules = read_in_data('datasets/sppnw41.txt')
best_solution, best_fitness = binary_genetic_algorithm('datasets/sppnw41.txt')
print_solution(best_solution, costs, coverage, rows)
feasible = is_feasible(best_solution, coverage, rows)
print(feasible)
print("\n")

def evaluate_algorithm(file_path, num_trials=30, max_generations=100):
    """Evaluates the genetic algorithm across multiple trials."""
    successful_runs = 0
    total_costs = []
    execution_times = []
    
    for trial in range(num_trials):
        print(f"Trial {trial+1}/{num_trials}...")
        start_time = time.time()
        solution, cost = binary_genetic_algorithm(file_path, max_generations=max_generations)
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








