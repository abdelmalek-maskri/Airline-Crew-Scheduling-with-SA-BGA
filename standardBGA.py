import numpy as np
import random
import time
import pandas as pd


def parse_problem(file_path):

    with open(file_path, 'r') as f:
        #first line: number of flight legs (rows) and round trips (cols)
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


def build_coverage_matrix(coverage, rows, cols):
    """
    Converts the list of coverage sets into a binary matrix.
    Each entry (i, j) in the matrix is 1 if schedule j covers flight i, otherwise 0.
    """
    coverage_matrix = np.zeros((rows, cols), dtype=int)
    for j in range(cols):
        for flight in coverage[j]:
            coverage_matrix[flight - 1, j] = 1
    return coverage_matrix


def initialize_population(population_size, chromosome_length, coverage, rows):
    """
    Initializes the population
    
    For each individual:
      - Schedules are added until all flights are covered.
      - Only non-overlapping schedules (with respect to current coverage) are selected.
    """
    population = np.zeros((population_size, chromosome_length), dtype=int)
    
    for individual in range(population_size):
        covered_flights = set()
        while len(covered_flights) < rows:
            available_schedules = [i for i in range(chromosome_length)
                                   if not covered_flights.intersection(coverage[i])]
            if not available_schedules:
                break  # No further schedules can be added without overlap
            chosen_schedule = np.random.choice(available_schedules)
            population[individual, chosen_schedule] = 1
            covered_flights.update(coverage[chosen_schedule])
    
    return population


def calculate_fitness(population, costs, coverage, rows):
    # The penalty is computed as the sum over flights of the absolute deviation

    cols = len(costs)
    
    # Create binary coverage matrix from list of coverage sets
    coverage_matrix = build_coverage_matrix(coverage, rows, cols)
    
    # Cost component: dot product of each individual with the cost vector
    fitness_values = np.dot(population, costs)
    
    # Compute the number of times each flight is covered for each individual
    coverage_counts = np.dot(coverage_matrix, population.T)  # Shape: (rows, pop_size)
    
    # Penalty: for each flight, add the absolute deviation from 1 (ideal coverage)
    penalties = np.sum(np.abs(coverage_counts - 1), axis=0)
    penalty_factor = 10000
    fitness_values = fitness_values + penalty_factor * penalties
    
    return fitness_values, penalties


def one_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    offspring2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return offspring1, offspring2

def mutate(offspring, mutation_probability, coverage, rows):
    """
    with a given probability, a new schedule is added to cover an uncovered flight,
    ensuring no overlapping coverage.
    """
    for i in range(len(offspring)):
        if random.random() < mutation_probability:
            #determine flights currently covered by this individual
            covered_flights = set()
            for schedule_id in range(len(offspring)):
                if offspring[schedule_id] == 1:
                    covered_flights.update(coverage[schedule_id])
            
            #determine flights that remain uncovered
            uncovered_flights = set(range(1, rows + 1)) - covered_flights
            
            #identify schedules that can cover an uncovered flight without overlapping
            available_schedules = [schedule_id for schedule_id in range(len(offspring))
                                   if not covered_flights.intersection(coverage[schedule_id])
                                   and uncovered_flights.intersection(coverage[schedule_id])]
            if available_schedules:
                chosen_schedule = np.random.choice(available_schedules)
                offspring[chosen_schedule] = 1
                covered_flights.update(coverage[chosen_schedule])
    
    return offspring


def tournament_selection(population, fitness_values, penalties, tournament_size=2):
    """
    Performs tournament selection to choose parents.
    
    In each tournament, individuals are compared first by penalty (feasibility)
    and then by fitness (cost), with the best being selected.
    """
    population_size = len(population)
    selected = []
    
    while len(selected) < population_size // 3:
        tournament_indices = random.sample(range(population_size), tournament_size)
        best_index = tournament_indices[0]
        for idx in tournament_indices[1:]:
            if penalties[idx] < penalties[best_index]:
                best_index = idx
            elif penalties[idx] == penalties[best_index] and fitness_values[idx] < fitness_values[best_index]:
                best_index = idx
        selected.append(population[best_index])
    
    return np.array(selected)


def binary_genetic_algorithm(file_path, population_size=100, max_generations=100, 
                             crossover_probability=0.85,
                             elite_percentage=0.1):

    costs, coverage, rows, cols = parse_problem(file_path)
    mutation_probability = 1 / cols
    
    #initialize the population based on non-overlapping schedules
    population = initialize_population(population_size, cols, coverage, rows)
    num_elite = max(1, int(population_size * elite_percentage))
    
    for generation in range(max_generations):
        fitness_values, penalties = calculate_fitness(population, costs, coverage, rows)
        
        #sort population based on fitness and penalties (lower is better)
        sorted_indices = np.lexsort((fitness_values, penalties))
        sorted_population = population[sorted_indices]
        elite_individuals = sorted_population[:num_elite].copy()
        
        #tournament selection for parents
        sorted_fitness = fitness_values[sorted_indices[:population_size]]
        sorted_penalties = penalties[sorted_indices[:population_size]]
        parents = tournament_selection(sorted_population[:population_size], sorted_fitness, sorted_penalties)
        
        #copy parents to form offspring
        offspring = parents.copy()
        
        #crossover
        for j in range(0, len(parents) - 1, 2):
            if random.random() < crossover_probability:
                offspring[j], offspring[j + 1] = one_point_crossover(parents[j], parents[j + 1])
        
        #mutate the offspring
        offspring = np.array([mutate(ind, mutation_probability, coverage, rows) for ind in offspring])
        
        #xombine elite individuals with new offspring
        population = np.vstack((elite_individuals, offspring))
        
        #ensure population size remains constant
        if len(population) < population_size:
            additional_individuals = initialize_population(population_size - len(population), cols, coverage, rows)
            population = np.vstack((population, additional_individuals))
    
    best_solution = population[0]
    best_fitness = np.dot(best_solution, costs)
    return best_solution, best_fitness


def is_feasible(solution, flight_coverage_list, num_flights):
    """
    Checks if every flight is covered exactly once.
    
    Returns True if each flight's coverage count is exactly one.
    """
    flight_coverage_count = [0] * num_flights
    for schedule_id in range(len(solution)):
        if solution[schedule_id] == 1:
            for flight in flight_coverage_list[schedule_id]:
                flight_coverage_count[flight - 1] += 1
    return all(count == 1 for count in flight_coverage_count)


def print_solution(solution, costs, coverage, rows):
    """
    Prints the final solution details.
    
    Displays:
      - Total number of selected schedules
      - Total cost
      - Details of each selected schedule and flight coverage
    """
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


def evaluate_algorithm(file_path, num_trials=30):
    """
    Evaluates the genetic algorithm over multiple trials.
    
    Returns statistics including:
      - Success Rate (%)
      - Average Cost
      - Standard Deviation of Cost
      - Average Execution Time (s)
    """
    successful_runs = 0
    total_costs = []
    execution_times = []
    
    for trial in range(num_trials):
        print(f"Trial {trial+1}/{num_trials}...")
        start_time = time.time()
        solution, cost = binary_genetic_algorithm(file_path)
        
        #retrieve the flight coverage list for feasibility checking
        _, coverage, rows, _ = parse_problem(file_path)
        feasible = is_feasible(solution, coverage, rows)
        
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



#run on a single dataset
file_path = 'datasets/sppnw42.txt'
costs, coverage, rows, cols = parse_problem(file_path)
print(f"Running standardBGA on {file_path}")
best_solution, best_cost = binary_genetic_algorithm(file_path)
print_solution(best_solution, costs, coverage, rows)

#evaluate 
#uncooment this to evaluate the algorithm over 30 runs
benchmark_files = ["datasets/sppnw41.txt", "datasets/sppnw42.txt", "datasets/sppnw43.txt"]
# benchmark_results = {file: evaluate_algorithm(file) for file in benchmark_files}
# df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
# print(df_results)
