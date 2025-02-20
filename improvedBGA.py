import numpy as np
import random
import time

def read_in_data(file_path):
    """Reads problem data from a file and returns the constraint matrix and column costs."""
    print(f"\n📂 Loading dataset: {file_path}")

    with open(file_path, 'r') as f:
        rows, cols = map(int, f.readline().split())
        con_matrix = np.zeros((rows, cols), dtype=int)  # Constraint matrix
        column_cost = np.zeros(cols, dtype=int)  # Cost per crew schedule

        for j in range(cols):
            line = list(map(int, f.readline().split()))
            column_cost[j] = line[0]  # Store column cost
            num_flights = line[1]
            flights = [f - 1 for f in line[2:2 + num_flights]]  # Convert to 0-based indexing
            con_matrix[flights, j] = 1  # Assign crew coverage

    print(f"✅ Loaded {rows} rows and {cols} columns")
    return con_matrix, column_cost

def initialize_population(num_ind, num_bit):
    """Generates initial population of binary individuals."""
    return np.random.randint(2, size=(num_ind, num_bit))

def calculate_fitness(pop, con_matrix, column_cost):
    """Calculates fitness and constraint violations."""
    fitness = np.dot(pop, column_cost)  # Total cost per individual
    constraint_violation = np.sum(np.maximum(0, 1 - con_matrix @ pop.T), axis=0)  # Uncovered flights
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

def mutate(offspring, mutation_prob):
    """Applies bit-flip mutation."""
    mask = np.random.rand(len(offspring)) < mutation_prob
    offspring[mask] = 1 - offspring[mask]
    return offspring

def heuristic_improvement_operator(individual, con_matrix, column_cost):
    """Improves individual feasibility."""
    for i in range(len(individual)):
        if individual[i] == 1:
            individual[i] = 0
            if np.sum(np.maximum(0, 1 - con_matrix @ individual)) > 0:
                individual[i] = 1  # Revert if infeasible
    return individual

def is_feasible(solution, con_matrix):
    """Checks feasibility of a solution."""
    return np.all(con_matrix @ solution >= 1)

def binary_genetic_algorithm(file_path, num_ind=40, max_iter=100, crossover_prob=0.85, mutation_prob=None):
    """Runs the Binary Genetic Algorithm (BGA)."""
    start_time = time.time()
    con_matrix, column_cost = read_in_data(file_path)
    num_bit = len(column_cost)
    mutation_prob = mutation_prob or (1 / num_bit)
    pop = initialize_population(num_ind, num_bit)
    
    for t in range(max_iter):
        parents = truncation_selection(pop, con_matrix, column_cost, num_ind // 3)
        offspring = parents.copy()
        
        for j in range(0, len(parents) - 1, 2):
            if random.random() < crossover_prob:
                offspring[j], offspring[j + 1] = one_point_crossover(parents[j], parents[j + 1])
        
        offspring = np.array([mutate(ind, mutation_prob) for ind in offspring])
        offspring = np.array([heuristic_improvement_operator(ind, con_matrix, column_cost) for ind in offspring])
        
        pop = np.vstack((pop, offspring))
        fitness, g_sum = calculate_fitness(pop, con_matrix, column_cost)
        sorted_idx = stochastic_ranking_sort(fitness, g_sum)
        pop = pop[sorted_idx[:num_ind]]
    
    best_fitness = fitness[sorted_idx[0]]
    best_solution = pop[0]
    print(f"\n✅ Completed in {time.time() - start_time:.2f} seconds.")
    print(f"🏅 Best fitness: {best_fitness}")
    return best_solution, best_fitness

# Example run
binary_genetic_algorithm('datasets/sppnw41.txt')
