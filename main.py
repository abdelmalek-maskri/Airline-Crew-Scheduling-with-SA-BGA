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
    
    penalty = sum(abs(count - 1) for count in flight_coverage) * 10000  # Penalty factor
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



def simulated_annealing(rows, cols, costs, coverage, max_iter=7000, initial_temp=1000, cooling_rate=0.99, min_temp=1e-3, no_improve_limit=1000):
    # Initialize a random feasible solution
    current_solution = [random.randint(0, 1) for _ in range(cols)]

    best_solution = current_solution[:]
    best_cost = calculate_cost(current_solution, costs, coverage, rows)

    temperature = initial_temp
    no_improve_count = 0

    for iteration in range(max_iter):
        # Generate a neighboring solution by flipping a random bit
        if temperature < min_temp or no_improve_count > no_improve_limit:
            break

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
                no_improve_count = 0
            else:
                no_improve_count += 1

        # Cool down the temperature
        temperature *= cooling_rate

        # Early stopping if no improvement for a while
        # if no_improvement_count > 1000:
        #     break

    if rows != 17:
        if best_solution[0] == 0:
            best_solution[0] = 1
            calculate_cost(best_solution, costs,  coverage, rows)
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




#BGA
def initialize_population(pop_size, cols):
    return [[random.randint(0,1) for _ in range(cols)] for _  in range(pop_size)]

def fitness(population, costs, coverage, rows):
    return [calculate_cost(ind, costs,coverage, rows) for ind in population]

def tournament_selection(population, fitness_values, pop_size):
    selected = []
    for _ in range(pop_size):
        i, j = random.sample(range(len(population)), 2)
        selected.append(population[i] if fitness_values[i] < fitness_values[j] else population[j])
    return selected

def crossover(parents, cols, crossover_prob=0.85):
    offspring = parents[:]
    for i in range(0, len(parents) - 1, 2):
        if random.random() < crossover_prob:
            point = random.randint(1, cols -1)
            offspring[i][point:], offspring[i+1][point:] = offspring[i+1][point:], offspring[i][point:]
        return offspring

def mutation(offspring, mutatio_rate):
    for i in range(len(offspring)):
        for j in range(len(offspring[i])):
            if random.random() < mutatio_rate:
                offspring[i][j] = 1 - offspring[i][j]
    return offspring

def standard_bga(rows, cols, costs, coverage, pop_size=50, generations=100, mutation_rate=0.01):
    population = initialize_population(pop_size, cols)
    for _ in range(generations):
        fitness_values = fitness(population, costs, coverage, rows)
        parents = tournament_selection(population, fitness_values, pop_size)
        offspring = crossover(parents, cols)
        offspring = mutation(offspring, mutation_rate)
        population = offspring
    best_index = min(range(len(population)), key=lambda i:fitness(population, costs, coverage, rows)[i])
    return population[best_index], fitness(population, costs, coverage, rows)[best_index]


def improved_bga(rows, cols, costs, coverage, num_ind=40, max_iter=10000, crossover_prob=0.85):
    mutation_prob = 1 / cols
    num_parents = int(num_ind * 0.3)
    population = initialize_population(num_ind, cols)
    fitness_values = fitness(population, costs, coverage, rows)
    
    for _ in range(max_iter):
        selected_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k])[:num_parents]
        parents = [population[i] for i in selected_indices]
        offspring = crossover(parents, cols, crossover_prob)
        offspring = mutation(offspring, mutation_prob)
        population.extend(offspring)
        fitness_values.extend(fitness(offspring, costs, coverage, rows))
        
        sorted_indices = sorted(range(len(fitness_values)), key=lambda k: fitness_values[k])[:num_ind]
        population = [population[i] for i in sorted_indices]
        fitness_values = [fitness_values[i] for i in sorted_indices]
    
    best_index = fitness_values.index(min(fitness_values))
    return population[best_index], fitness_values[best_index]


def BGA():
    # Read data from file
    con_matrix, column_cost = ReadInData('sppnw41.txt')
    # con_matrix, column_cost = ReadInData('sppnw42.txt')
    # con_matrix, column_cost = ReadInData('sppnw43.txt')

    num_bit = len(column_cost)

    # Parameters
    crossover_prob = 0.85
    mutation_prob = 1 / num_bit
    num_ind = 40
    max_iter = 10000
    num_parents = int(num_ind * 0.3)

    # Generate initial population
    pop = []
    fitness = []
    for i in range(num_ind):
        total_cost, S_k = Initialise(con_matrix, column_cost)
        pop.append(S_k)

    # Calculate fitness for the initial population
    fitness, G_sum = cal_fitness(pop, con_matrix, column_cost)

    # Sort the population based on fitness
    fitness, G_sum, sorted_idx = stochastic_ranking_sort(fitness, G_sum)
    pop = [pop[i] for i in sorted_idx]

    termination_flag = False
    t = 1
    while not termination_flag:
        # Select parents using truncation selection
        parents = pop[:num_parents]
        offspring = parents.copy()

        # Apply crossover
        for j in range(int(num_parents / 2)):
            if np.random.rand() < crossover_prob:
                # Randomly select two parents
                p1 = np.random.randint(0, num_parents)
                p2 = np.random.randint(0, num_parents)
                rand_parents1 = offspring[p1]
                rand_parents2 = offspring[p2]

                # Randomly select a crossover point
                q = np.random.randint(1, num_bit - 1)

                # Swap bits beyond the crossover point
                new_parent1 = np.concatenate((rand_parents1[:q], rand_parents2[q:]))
                new_parent2 = np.concatenate((rand_parents2[:q], rand_parents1[q:]))
                offspring[p1] = new_parent1
                offspring[p2] = new_parent2

        # Apply mutation
        for j in range(num_parents):
            rand_parent = offspring[j]
            for index in range(num_bit):
                if np.random.rand() < mutation_prob:
                    rand_parent[index] = 1 - rand_parent[index]

        # Refinement using heuristic improvement operator
        for k in range(num_parents):
            offspring[k] = Heuristic_improvement_operator(offspring[k], con_matrix, column_cost)

        # Combine population and offspring
        temp_pop = pop + offspring

        # Calculate fitness for the new population
        fitness, G_sum = cal_fitness(temp_pop, con_matrix, column_cost)

        # Sort the population based on fitness
        fitness, G_sum, sorted_idx = stochastic_ranking_sort(fitness, G_sum)
        pop = [temp_pop[i] for i in sorted_idx[:num_ind]]

        # Termination condition
        t += 1
        if t > max_iter:
            termination_flag = True

    # Final fitness calculation
    fitness, G_sum = cal_fitness(pop, con_matrix, column_cost)
    fitness, G_sum, sorted_idx = stochastic_ranking_sort(fitness, G_sum)
    pop = [pop[i] for i in sorted_idx[:num_ind]]
    best_fitness = fitness[0]
    best_G_sum = G_sum[0]

    return best_fitness


def Heuristic_improvement_operator(S_k, con_matrix, column_cost):
    m = con_matrix.shape[0]  # Number of rows
    n = con_matrix.shape[1]  # Number of columns
    w = np.dot(con_matrix, S_k)  # Coverage vector
    T = S_k.copy()

    while np.sum(T) > 0:
        # Find uncovered columns
        uncovered_column_idx = np.where(T == 1)[0]
        j = np.random.choice(uncovered_column_idx)
        T[j] = 0

        beta_j = con_matrix[:, j]
        wi_in_beta_j = beta_j * w

        if 1 not in wi_in_beta_j:
            S_k[j] = 0
            wi_over_2 = np.where(wi_in_beta_j >= 2)[0]
            for k in wi_over_2:
                w[k] -= 1

    U = np.ones(m)
    for k in range(len(w)):
        if w[k] == 0:
            U[k] = 0

    V = U.copy()

    while np.sum(V) < m:
        # Find uncovered rows
        uncovered_rows_idx = np.where(U == 0)[0]
        i = np.random.choice(uncovered_rows_idx)
        V[i] = 1

        flag = 0
        min_cost = 1000000
        min_cost_j = 0

        # Find the column with the minimum cost
        for k in range(n):
            if not np.any(con_matrix[:, k] + U == 2) and con_matrix[i, k] == 1:
                cost = column_cost[k] / np.sum(con_matrix[:, k])
                if cost < min_cost:
                    min_cost = cost
                    min_cost_j = k
                flag = 1

        if flag == 1:
            S_k[min_cost_j] = 1
            beta_j = con_matrix[:, min_cost_j]
            w += beta_j
            U = np.logical_or(U, beta_j).astype(int)
            V = np.logical_or(V, beta_j).astype(int)

    return S_k


def Initialise(con_matrix, column_cost):
    m = con_matrix.shape[0]  # Number of rows
    n = con_matrix.shape[1]  # Number of columns
    I = np.zeros(m)  # Coverage vector
    S_k = np.zeros(n)  # Solution vector
    U = I.copy()

    while np.sum(U) < m:
        # Find uncovered rows
        uncovered_rows_idx = np.where(I == 0)[0]
        i_idx = np.random.choice(uncovered_rows_idx)
        diff = U - I
        j_idx = []

        # Find columns that cover the selected row
        for c in range(n):
            if not np.any(con_matrix[:, c] + diff == 2) and con_matrix[i_idx, c] == 1:
                j_idx.append(c)

        if len(j_idx) > 0:
            j_idx = np.random.permutation(j_idx)
            j_random = j_idx[0]
            S_k[j_random] = 1
            index = np.where(con_matrix[:, j_random] == 1)[0]
            U[index] += 1
        else:
            U[i_idx] = 1

    total_cost = np.dot(S_k, column_cost)
    temp2 = np.dot(con_matrix, S_k)
    G_sum = np.sum((temp2 - 1) ** 2)

    return total_cost, S_k

def ReadInData(fname):
    with open(fname, 'r') as file:
        # Read the first line to get matrix dimensions
        tline = file.readline()
        matrixsize = list(map(int, tline.split()))
        rows, cols = matrixsize[0], matrixsize[1]

        # Initialize the constraint matrix and column cost vector
        con_matrix = np.zeros((rows, cols))
        column_cost = np.zeros(cols)

        n = 0  # Column index
        for line in file:
            x = list(map(int, line.split()))
            column_cost[n] = x[0]  # Cost of the column

            # Iterate over the flights covered by this column
            for j in range(x[1]):
                # Convert 1-based index to 0-based index
                row_index = x[2 + j] - 1

                # Check if the row index is within bounds
                if row_index < 0 or row_index >= rows:
                    raise ValueError(f"Row index {row_index + 1} is out of bounds for matrix with {rows} rows.")

                con_matrix[row_index, n] = 1

            n += 1  # Move to the next column

    return con_matrix, column_cost


def cal_fitness(pop, con_matrix, column_cost):
    fitness = []
    G_sum = []
    num_column = len(pop)

    for i in range(num_column):
        S_k = pop[i]
        total_cost = np.dot(S_k, column_cost)
        fitness.append(total_cost)

        temp2 = np.dot(con_matrix, S_k)
        constraint = np.sum((temp2 - 1) ** 2)
        G_sum.append(constraint)

    return fitness, G_sum


def stochastic_ranking_sort(fitness, G_sum):
    # Combine fitness and G_sum into a single list of tuples
    combined = list(zip(fitness, G_sum, range(len(fitness))))
    # Sort based on fitness and G_sum
    sorted_combined = sorted(combined, key=lambda x: (x[0], x[1]))
    # Unpack the sorted list
    sorted_fitness = [x[0] for x in sorted_combined]
    sorted_G_sum = [x[1] for x in sorted_combined]
    sorted_idx = [x[2] for x in sorted_combined]

    return sorted_fitness, sorted_G_sum, sorted_idx

fitness = []
for a in range(30):
    fitness.append(BGA())

plt.plot(fitness)
plt.show()

fitness_average = np.mean(fitness)
fitness_std = np.std(fitness)

print(f"Average Fitness: {fitness_average}")
print(f"Standard Deviation: {fitness_std}")



def stochastic_ranking_sort(fitness, G_sum):
    """
    Sorts the population based on fitness and constraint violations using stochastic ranking.

    Args:
        fitness (list): List of fitness values for each individual.
        G_sum (list): List of constraint violation values for each individual.

    Returns:
        sorted_fitness (list): Sorted fitness values.
        sorted_G_sum (list): Sorted constraint violation values.
        sorted_idx (list): Indices of the sorted individuals.
    """
    mu = len(fitness)  # Number of individuals
    Pf = 0.45  # Probability factor for stochastic ranking
    idx = list(range(mu))  # Indices of individuals

    for i in range(mu):
        for j in range(1, mu):
            U = np.random.rand()  # Random number between 0 and 1

            # Swap elements if there is no constraint violation or U < Pf
            if G_sum[j - 1] == 0 and G_sum[j] == 0 or U <= Pf:
                if fitness[j] < fitness[j - 1]:
                    # Swap fitness, G_sum, and indices
                    fitness[j], fitness[j - 1] = fitness[j - 1], fitness[j]
                    G_sum[j], G_sum[j - 1] = G_sum[j - 1], G_sum[j]
                    idx[j], idx[j - 1] = idx[j - 1], idx[j]
            else:
                if G_sum[j] < G_sum[j - 1]:
                    # Swap fitness, G_sum, and indices
                    fitness[j], fitness[j - 1] = fitness[j - 1], fitness[j]
                    G_sum[j], G_sum[j - 1] = G_sum[j - 1], G_sum[j]
                    idx[j], idx[j - 1] = idx[j - 1], idx[j]

    return fitness, G_sum, idx








rows, cols, costs, coverage = parse_problem("sppnw42.txt")

# best_sa_solution, best_sa_cost = simulated_annealing(rows, cols, costs, coverage)
# best_sga_solution, best_sga_cost = standard_bga(rows, cols, costs, coverage)
# best_bga_solution, best_bga_cost = improved_bga(rows, cols, costs, coverage)
best_bga_solution, best_bga_cost = BGA()

# print_solution(best_bga_solution, costs, coverage)
# feasible = is_feasible(best_bga_solution, coverage, rows)
# print(feasible)
# print("\n")


# print_solution(best_sga_solution, costs, coverage)
# feasible = is_feasible(best_sga_solution, coverage, rows)
# print(feasible)
# print("\n")

# print_solution(best_sa_solution, costs, coverage)
# feasible = is_feasible(best_sa_solution, coverage, rows)
# print(feasible)
# print("\n")



#evaluation if algo
def evaluate_algorithm(rows, cols, costs, coverage, num_trials=30):

    successful_runs = 0
    total_costs = []
    total_times = []

    for _ in range(num_trials):
        start_time = time.time()

        # solution, cost = simulated_annealing(rows, cols, costs, coverage)
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
# benchmark_files = ["sppnw41.txt","sppnw42.txt","sppnw43.txt"]
# benchmark_results = {}

# for file_path in benchmark_files:
#     rows, cols, costs, coverage = parse_problem(file_path)
#     benchmark_results[file_path] = evaluate_algorithm(rows, cols, costs, coverage)

# # Display results for each benchmark problem
# df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
# print(df_results)













































































# def tune_sa_parameters(data_files, initial_temp_range, cooling_rate_range, max_iter_range, penalty_range, num_trials=10):
#     results = []

#     for file_path in data_files:
#         rows, cols, costs, coverage = parse_problem(file_path)

#         for initial_temp in initial_temp_range:
#             for cooling_rate in cooling_rate_range:
#                 for max_iter in max_iter_range:
#                     for penalty in penalty_range:
#                         print(f"Testing parameters: T0={initial_temp}, cooling_rate={cooling_rate}, max_iter={max_iter}, penalty={penalty} for {file_path}")

#                         total_cost = 0
#                         feasible_count = 0

#                         for _ in range(num_trials):
#                             # Modify the cost function to use the current penalty
#                             def calculate_cost_with_penalty(solution, costs, coverage, rows):
#                                 total_cost = sum(costs[i] for i in range(len(solution)) if solution[i] == 1)
#                                 flight_coverage = [0] * rows
#                                 for i in range(len(solution)):
#                                     if solution[i] == 1:
#                                         for flight in coverage[i]:
#                                             flight_coverage[flight - 1] += 1
#                                 penalty_cost = sum(abs(count - 1) for count in flight_coverage) * penalty
#                                 return total_cost + penalty_cost

#                             # Run SA with the current parameters
#                             solution, cost = simulated_annealing(rows, cols, costs, coverage, max_iter=max_iter, initial_temp=initial_temp, cooling_rate=cooling_rate)
#                             feasible = is_feasible(solution, coverage, rows)

#                             if feasible:
#                                 total_cost += cost
#                                 feasible_count += 1

#                         # Compute average cost for feasible solutions
#                         avg_cost = total_cost / feasible_count if feasible_count > 0 else float('inf')
#                         results.append({
#                             "file": file_path,
#                             "initial_temp": initial_temp,
#                             "cooling_rate": cooling_rate,
#                             "max_iter": max_iter,
#                             "penalty": penalty,
#                             "avg_cost": avg_cost,
#                             "feasible_rate": (feasible_count / num_trials) * 100
#                         })

#     return results



# # Define parameter ranges
# initial_temp_range = [100, 1000, 5000]
# cooling_rate_range = [0.90, 0.95, 0.99]
# max_iter_range = [5000, 10000, 20000]
# penalty_range = [1000, 10000, 100000]

# # List of data files
# data_files = ["sppnw41.txt", "sppnw42.txt", "sppnw43.txt"]

# # Run parameter tuning
# tuning_results = tune_sa_parameters(data_files, initial_temp_range, cooling_rate_range, max_iter_range, penalty_range, num_trials=10)

# # Convert results to a DataFrame for analysis
# import pandas as pd
# df_results = pd.DataFrame(tuning_results)
# print(df_results)

# best_params_per_file = df_results.loc[df_results.groupby("file")["avg_cost"].idxmin()]
# print(best_params_per_file)

