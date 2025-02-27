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

def pseudo_random_initialization(population_size, chromosome_length, coverage, rows):
    """
    Initialize population using a pseudo-random method that selects columns based on randomly chosen rows.
    """
    population = np.zeros((population_size, chromosome_length), dtype=int)
    
    # Create a reverse mapping: for each row, which columns cover it
    row_to_columns = [[] for _ in range(rows + 1)]  # +1 because rows are 1-indexed
    for column_idx, covered_rows in enumerate(coverage):
        for row in covered_rows:
            row_to_columns[row].append(column_idx)
    
    for individual_idx in range(population_size):
        # Start with an empty solution
        solution = np.zeros(chromosome_length, dtype=int)
        
        # Set of uncovered rows (1-indexed)
        uncovered_rows = set(range(1, rows + 1))
        
        # Continue until all rows are covered or no more valid columns can be found
        while uncovered_rows:
            # Randomly select an uncovered row
            if not uncovered_rows:
                break
            
            selected_row = random.choice(list(uncovered_rows))
            
            # Find columns that can cover this row
            valid_columns = row_to_columns[selected_row]
            
            if valid_columns:
                # Randomly select a column
                selected_column = random.choice(valid_columns)
                
                # Add this column to the solution
                solution[selected_column] = 1
                
                # Update uncovered rows
                rows_covered_by_column = coverage[selected_column]
                uncovered_rows -= rows_covered_by_column
            else:
                # If no valid column can cover this row, simply remove it and continue
                uncovered_rows.remove(selected_row)
        
        population[individual_idx] = solution
    
    return population

def heuristic_improvement_operator(solution, costs, coverage, rows, drop_factor=1.0, add_factor=1.0):
    """
    Apply the heuristic improvement operator with DROP and ADD procedures.
    
    Args:
        solution: Binary solution vector
        costs: Array of costs for each column
        coverage: List of sets, each set containing the rows covered by that column
        rows: Total number of rows to cover
        drop_factor: Controls aggressiveness of DROP procedure (higher = more aggressive)
        add_factor: Controls how cost is weighted in ADD procedure (higher = more cost sensitive)
        
    Returns:
        Improved solution
    """
    improved_solution = solution.copy()
    
    # Create a coverage count for each row
    row_coverage_count = [0] * (rows + 1)  # +1 because rows are 1-indexed
    for column_idx in range(len(improved_solution)):
        if improved_solution[column_idx] == 1:
            for row in coverage[column_idx]:
                row_coverage_count[row] += 1
    
    # ========== DROP PROCEDURE ==========
    # Find over-covered rows (covered by more than one column)
    over_covered_rows = [row for row in range(1, rows + 1) if row_coverage_count[row] > 1]
    
    # Apply drop factor - higher drop_factor increases probability of removing redundant columns
    if random.random() < drop_factor:
        # For each over-covered row, randomly remove columns until it's covered exactly once
        for row in over_covered_rows:
            # Find all columns covering this row that are currently in the solution
            covering_columns = [col_idx for col_idx in range(len(improved_solution)) 
                              if improved_solution[col_idx] == 1 and row in coverage[col_idx]]
            
            # Sort by cost in descending order (try to remove expensive columns first if drop_factor > 1.0)
            if drop_factor > 1.0:
                covering_columns.sort(key=lambda col: costs[col], reverse=True)
            else:
                # Randomly shuffle the columns to ensure random removal
                random.shuffle(covering_columns)
            
            # Keep removing columns until this row is covered exactly once
            for col_idx in covering_columns[1:]:  # Skip the first column to keep at least one
                # Check if removing this column would leave all rows covered
                can_remove = True
                for covered_row in coverage[col_idx]:
                    if row_coverage_count[covered_row] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    improved_solution[col_idx] = 0
                    # Update coverage counts
                    for covered_row in coverage[col_idx]:
                        row_coverage_count[covered_row] -= 1
    
    # ========== ADD PROCEDURE ==========
    # Find under-covered rows (not covered by any column)
    under_covered_rows = [row for row in range(1, rows + 1) if row_coverage_count[row] == 0]
    
    if under_covered_rows:
        # Create a reverse mapping: for each row, which columns cover it
        row_to_columns = [[] for _ in range(rows + 1)]
        for column_idx, covered_rows in enumerate(coverage):
            for row in covered_rows:
                row_to_columns[row].append(column_idx)
        
        # For each under-covered row, add a column that covers it while considering cost effectiveness
        for row in under_covered_rows:
            # Find columns that cover this row
            candidate_columns = row_to_columns[row]
            
            if candidate_columns:
                # Calculate cost effectiveness: value = newly covered rows / cost
                # Higher add_factor gives more weight to cost considerations
                column_values = []
                for col in candidate_columns:
                    new_rows_covered = sum(1 for r in coverage[col] if row_coverage_count[r] == 0)
                    # Avoid division by zero
                    if costs[col] == 0:
                        column_values.append((col, float('inf')))
                    else:
                        value = (new_rows_covered ** add_factor) / costs[col]
                        column_values.append((col, value))
                
                # Sort by value in descending order
                column_values.sort(key=lambda x: x[1], reverse=True)
                
                # Select the best column
                best_column = column_values[0][0]
                improved_solution[best_column] = 1
                
                # Update coverage counts
                for covered_row in coverage[best_column]:
                    row_coverage_count[covered_row] += 1
    
    return improved_solution


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

# def one_point_crossover(parent1, parent2):
#     """Performs one-point crossover between two parent solutions."""
#     crossover_point = random.randint(1, len(parent1) - 1)  
#     return np.concatenate((parent1[:crossover_point], parent2[crossover_point:])), \
#            np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))

def uniform_crossover(parent1, parent2, mix_ratio=0.5):
    """
    Performs uniform crossover between two parent solutions.
    
    Args:
        parent1, parent2: The parent solutions
        mix_ratio: Probability of inheriting from parent1 (0.5 = equal chance)
        
    Returns:
        Two offspring solutions
    """
    mask = np.random.random(len(parent1)) < mix_ratio
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)
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

def binary_genetic_algorithm_with_stochastic_ranking(file_path, population_size=150, max_generations=100, 
                                                     crossover_probability=0.85, mutation_probability=None, p_f=0.45,
                                                     apply_heuristic_improvement=True):
    """
    Binary Genetic Algorithm with Stochastic Ranking for the Set Covering Problem.
    """
    costs, coverage, rows, cols = read_in_data(file_path)
    
    mutation_probability = mutation_probability or (1 / cols)
    
    # Use pseudo-random initialization
    population = pseudo_random_initialization(population_size, cols, coverage, rows)
    
    # Apply heuristic improvement to initial population if specified
    if apply_heuristic_improvement:
        population = np.array([heuristic_improvement_operator(ind, costs, coverage, rows) for ind in population])
    
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
                offspring[j], offspring[j + 1] = uniform_crossover(parents[j], parents[j + 1])
        
        # Apply mutation
        offspring = np.array([mutate(ind, mutation_probability, coverage, rows) for ind in offspring])
        
        # Apply heuristic improvement to offspring if specified
        if apply_heuristic_improvement:
            offspring = np.array([heuristic_improvement_operator(ind, costs, coverage, rows) for ind in offspring])
        
        # Update population with offspring
        population = np.vstack((population, offspring))

    # Get the best solution after all generations
    final_objective_values, final_constraint_violations = calculate_fitness(population, costs, coverage, rows)
    sorted_indices = stochastic_ranking(population, final_objective_values, final_constraint_violations, p_f)
    best_solution = population[sorted_indices[0]]
    
    # Apply a final heuristic improvement to the best solution
    if apply_heuristic_improvement:
        best_solution = heuristic_improvement_operator(best_solution, costs, coverage, rows)
    
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