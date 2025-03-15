import numpy as np
import time
import csv

def import_data_numpy(filename):
    """Imports data from a text file for an airline scheduling problem using NumPy.

    Args:
        filename: The name of the file to import.

    Returns:
        A tuple containing:
            - n_legs: The number of flight legs (rows).
            - n_round_trips: The number of round trips (columns).
            - costs_vector: A NumPy array of column costs.
            - coverage_matrix: A NumPy binary matrix where (i, j) = 1 if round trip j covers leg i.
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            n_legs, n_round_trips = map(int, lines[0].split())
            costs_vector = np.zeros(n_round_trips, dtype=int)
            coverage_matrix = np.zeros((n_legs, n_round_trips), dtype=int)
            line_index = 1
            for j in range(n_round_trips):
                column_data = list(map(int, lines[line_index].split()))
                costs_vector[j] = column_data[0]
                legs = column_data[2:]
                for leg in legs:
                    coverage_matrix[leg - 1, j] = 1
                line_index += 1
        return n_legs, n_round_trips, costs_vector, coverage_matrix
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None, None, None

def pseudo_random_initialization(n_legs, n_round_trips, coverage_matrix, pop_size=100):
    """
    Generates an initial population for the Set Partitioning Problem (SPP).
    """
    population = np.zeros((pop_size, n_round_trips), dtype=int)
    for k in range(pop_size):
        individual = np.zeros(n_round_trips, dtype=int)
        U = set(range(n_legs))
        while U:
            i = np.random.choice(list(U))
            valid_columns = np.where(coverage_matrix[i] == 1)[0]
            valid_columns = [j for j in valid_columns if not any(coverage_matrix[list(set(range(n_legs)) - U), j])]
            if valid_columns:
                j = np.random.choice(valid_columns)
                individual[j] = 1
                U -= set(np.where(coverage_matrix[:, j] == 1)[0])
            else:
                U.remove(i)
        population[k] = individual
    return population

def compute_fitness_and_feasibility_individual(individual, costs_vector, coverage_matrix):
    """
    Computes the fitness (cost) and feasibility (constraint violations) of an individual.
    """
    fitness = np.dot(costs_vector, individual)
    coverage = np.dot(coverage_matrix, individual)
    infeasibility = np.sum((coverage == 0) | (coverage > 1))
    return fitness, infeasibility

def compute_fitness_and_feasibility_population(population, costs_vector, coverage_matrix):
    """
    Efficiently computes fitness and feasibility for the entire population.
    """
    fitness = np.dot(population, costs_vector)
    coverage = np.dot(coverage_matrix, population.T)
    feasibility = np.sum((coverage == 0) | (coverage > 1), axis=0)
    return list(zip(fitness, feasibility))

def tournament_selection(population, costs_vector, coverage_matrix, num_parents, tournament_size):
    """
    Performs tournament selection on a stochastically ranked population.
    """
    parents = []
    population_size = len(population)
    for _ in range(num_parents):
        tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_scores = [
            compute_fitness_and_feasibility_individual(ind, costs_vector, coverage_matrix)
            for ind in tournament_individuals
        ]
        best_index = min(
            range(tournament_size),
            key=lambda i: (tournament_scores[i][1], tournament_scores[i][0])
        )
        parents.append(tournament_individuals[best_index])
    return np.array(parents)

def uniform_crossover(parents, crossover_rate=0.8):
    """
    Performs uniform crossover on a set of parents to generate offspring.
    """
    num_parents, num_genes = parents.shape
    offspring = []
    for i in range(0, num_parents - 1, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        if np.random.rand() < crossover_rate:
            mask = np.random.randint(0, 2, num_genes)
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        else:
            child1 = parent1.copy()
            child2 = parent2.copy()
        offspring.extend([child1, child2])
    if num_parents % 2 != 0:
        offspring.append(parents[-1].copy())
    return np.array(offspring)

def heuristic_improvement(individual, costs_vector, coverage_matrix):
    """
    Heuristic improvement operator for the Set Partitioning Problem (SPP).
    """
    improved_individual = individual.copy()
    coverage = np.dot(coverage_matrix, improved_individual)
    w_i = coverage.copy()
    T = np.where(improved_individual == 1)[0].tolist()
    while T:
        j = np.random.choice(T)
        T.remove(j)
        if all(w_i[coverage_matrix[:, j] == 1] >= 2):
            improved_individual[j] = 0
            for i in np.where(coverage_matrix[:, j] == 1)[0]:
                w_i[i] -= 1
    U = np.where(w_i == 0)[0].tolist()
    while U:
        i = np.random.choice(U)
        U.remove(i)
        candidate_columns = [
            (j, costs_vector[j] / np.sum(coverage_matrix[:, j]))
            for j in np.where(coverage_matrix[i, :] == 1)[0]
            if np.all(coverage_matrix[:, j] == 0) or np.all(w_i + coverage_matrix[:, j] <= 1)
        ]
        if candidate_columns:
            j = min(candidate_columns, key=lambda x: x[1])[0]
            improved_individual[j] = 1
            for i in np.where(coverage_matrix[:, j] == 1)[0]:
                w_i[i] += 1
            U = [i for i in U if w_i[i] == 0]
    return improved_individual

def proportional_bit_flip_mutation(offspring, n_round_trips, mutation_chance):
    """
    Applies proportional bit-flip mutation where each gene has a 1/n_round_trips chance of flipping.
    """
    mutation_prob = mutation_chance / n_round_trips
    mutation_mask = np.random.rand(*offspring.shape) < mutation_prob
    mutated_offspring = np.copy(offspring)
    mutated_offspring[mutation_mask] ^= 1
    return mutated_offspring

def stochastic_ranking(population, costs_vector, coverage_matrix, Pf=0.45, max_iterations=1000):
    """
    Applies Stochastic Ranking to sort the population based on a balance of fitness and feasibility.
    """
    fitness_feasibility = compute_fitness_and_feasibility_population(population, costs_vector, coverage_matrix)
    population_size = len(population)
    for _ in range(max_iterations):
        swapped = False
        for i in range(population_size - 1):
            fit_i, feas_i = fitness_feasibility[i]
            fit_i1, feas_i1 = fitness_feasibility[i + 1]
            u = np.random.rand()
            if (feas_i == feas_i1 == 0) or (u <= Pf):
                if fit_i > fit_i1:
                    population[[i, i + 1]] = population[[i + 1, i]]
                    fitness_feasibility[i], fitness_feasibility[i + 1] = fitness_feasibility[i + 1], fitness_feasibility[i]
                    swapped = True
            else:
                if feas_i > feas_i1:
                    population[[i, i + 1]] = population[[i + 1, i]]
                    fitness_feasibility[i], fitness_feasibility[i + 1] = fitness_feasibility[i + 1], fitness_feasibility[i]
                    swapped = True
        if not swapped:
            break
    return population

def improved_bga(n_legs, n_round_trips, costs_vector, coverage_matrix,
                 population_size, generations, tournament_size,
                 crossover_rate, base_mutation_rate, stochastic_ranking_pf,
                 elitism):
    """
    Implements the Improved Binary Genetic Algorithm (BGA) for the Set Partitioning Problem.
    """
    population = pseudo_random_initialization(n_legs, n_round_trips, coverage_matrix, population_size)
    for generation in range(generations):
        population = stochastic_ranking(population, costs_vector, coverage_matrix, Pf=stochastic_ranking_pf)[:population_size]
        num_elite_parents = int(len(population) * elitism)
        elite_parents = population[:num_elite_parents]
        selected_parents = tournament_selection(elite_parents, costs_vector, coverage_matrix, num_elite_parents, tournament_size)
        offspring = uniform_crossover(selected_parents, crossover_rate)
        mutated_offspring = proportional_bit_flip_mutation(offspring, n_round_trips, base_mutation_rate)
        improved_offspring = np.array([
            heuristic_improvement(ind, costs_vector, coverage_matrix)
            for ind in mutated_offspring
        ])
        population = np.vstack((population, improved_offspring))
    best_individual = population[0]
    best_solution = np.where(best_individual == 1)[0]
    best_cost, best_feasibility = compute_fitness_and_feasibility_individual(best_individual, costs_vector, coverage_matrix)
    return best_solution, best_cost, best_feasibility

def print_detailed_solution(solution, cost, feasibility, costs_vector, coverage_matrix, n_round_trips):
    """
    Prints the detailed genetic algorithm solution.
    """
    total_selected = len(solution)
    total_cost = cost
    print("==== Genetic Algorithm Solution ====")
    print(f"Total Selected Crew Schedules: {total_selected}")
    print(f"Total Cost: {total_cost}")
    print()
    print("Selected Crew Schedules:")
    for j in solution:
        schedule_cost = costs_vector[j]
        # Find flight indices (add 1 for 1-indexing)
        flights = list((np.where(coverage_matrix[:, j] == 1)[0] + 1))
        print(f"- Schedule {j+1}: Cost = {schedule_cost}, Covers Flights: {flights}")
    print("====================================")
    # Compute flight coverage
    individual = np.zeros(n_round_trips, dtype=int)
    individual[solution] = 1
    flight_coverage = np.dot(coverage_matrix, individual)
    print()
    print("Flight coverage: ", flight_coverage.tolist())
    print("Solution is feasible:", feasibility == 0)
    print()

# List of dataset filenames
filenames = ["datasets/sppnw41.txt", "datasets/sppnw42.txt", "datasets/sppnw43.txt"]

# Prepare to collect summary results for CSV export
summary_results = []

# Process each dataset
for dataset_index, filename in enumerate(filenames):
    n_legs, n_round_trips, costs_vector, coverage_matrix = import_data_numpy(filename)
    dataset_costs = []
    dataset_times = []
    success_count = 0

    # For the first dataset, run once and print detailed solution output
    if dataset_index == 0:
        best_solution, best_cost, best_feasibility = improved_bga(
            n_legs, n_round_trips, costs_vector, coverage_matrix,
            population_size=100, generations=100, tournament_size=2,
            crossover_rate=0.70, base_mutation_rate=0.8, stochastic_ranking_pf=0.47,
            elitism=1
        )
        print_detailed_solution(best_solution, best_cost, best_feasibility, costs_vector, coverage_matrix, n_round_trips)

    # Run 30 trials for statistics (printing only trial numbers)
    for j in range(30):
        print(f"Trial {j+1}/30...")
        start_time = time.time()
        solution, cost, feasibility = improved_bga(
            n_legs, n_round_trips, costs_vector, coverage_matrix,
            population_size=100, generations=100, tournament_size=2,
            crossover_rate=0.70, base_mutation_rate=0.8, stochastic_ranking_pf=0.47,
            elitism=1
        )
        elapsed_time = time.time() - start_time
        dataset_costs.append(cost)
        dataset_times.append(elapsed_time)
        if feasibility == 0:
            success_count += 1

    success_rate = (success_count / 30) * 100
    avg_cost = np.mean(dataset_costs)
    std_cost = np.std(dataset_costs)
    avg_time = np.mean(dataset_times)
    
    # Save the results for CSV export and later summary printing
    summary_results.append([filename, success_rate, avg_cost, std_cost, avg_time])

# Print summary table
print("\n                      Success Rate (%)  Average Cost  Standard Deviation  Average Execution Time (s)")
for result in summary_results:
    print(f"{result[0]:30} {result[1]:10.1f} {result[2]:15.6f} {result[3]:20.6f} {result[4]:30.6f}")

# Export summary results to CSV
with open("summary.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Dataset", "Success Rate (%)", "Average Cost", "Standard Deviation", "Average Execution Time (s)"])
    writer.writerows(summary_results)

print("\nSummary exported to summary.csv")
