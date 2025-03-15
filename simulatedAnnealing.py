import random
import math
import time
import numpy as np
import pandas as pd
from collections import Counter

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
            
    return rows, cols, costs, coverage
    
# rows, cols,costs, coverage = parse_problem('sppnw41.txt'
# print(f"rows: ", {rows})
# print(f"cols: ", {cols})
# print("costs: ", costs)
# print("coverage: ",coverage)

def convert_coverage_to_matrix(coverage, rows, cols):
    #convert a list-of-sets (coverage) to a NumPy binary matrix.
    #each column j of the returned matrix has a 1 in row i if (i+1) is in coverage[j].
    mat = np.zeros((rows, cols), dtype=int)
    for j in range(cols):
        for r in coverage[j]:
            mat[r - 1, j] = 1
    return mat

def convert_indices_to_binary(selected_indices, cols):
    #given an array of selected round trip indices, return a binary vector (list) of length cols.
    sol = [0] * cols
    for i in selected_indices:
        sol[i] = 1
    return sol

def initialise_valid_solution(coverage_matrix):
    """
    Generates a random valid initial solution ensuring all flight legs are covered exactly once.

    Returns:
        A NumPy array of selected round trip indices.
    """
    rows, cols = coverage_matrix.shape
    selected_round_trips = np.zeros(cols, dtype=bool)
    covered_rows = np.zeros(rows, dtype=bool)
    required_rows = np.ones(rows, dtype=bool)

    #step 1:select a random round trip to start
    first_round_trip = random.randint(0, cols - 1)
    selected_round_trips[first_round_trip] = True
    covered_rows = (coverage_matrix @ selected_round_trips) > 0

    #step 2:add more round trips until all legs are covered
    while not np.array_equal(covered_rows, required_rows):
        available_round_trips = np.where(np.all(coverage_matrix[covered_rows, :] == 0, axis=0))[0]
        if available_round_trips.size == 0:
            #restart if no valid replacement is found
            return initialise_valid_solution(coverage_matrix)
        chosen_round_trip = np.random.choice(available_round_trips)
        selected_round_trips[chosen_round_trip] = True
        covered_rows = (coverage_matrix @ selected_round_trips) > 0

    selected_indices = np.where(selected_round_trips)[0]
    # print("initial valid solution indices:", selected_indices)
    return selected_indices

def generate_neighbor_solution(current_solution_indices, coverage_matrix):
    """
    Generates a neighboring solution by removing 3-4 round trips at random and then
    adding valid replacements until all legs are covered.
    """
    rows, cols = coverage_matrix.shape
    # Convert current solution indices into a boolean selection mask
    selection_mask = np.zeros(cols, dtype=bool)
    selection_mask[current_solution_indices] = True
    covered_rows = (coverage_matrix @ selection_mask) > 0
    required_rows = np.ones(rows, dtype=bool)

    #remove 3-4 round trips at random
    removable_neighbours = np.where(selection_mask)[0]
    num_replacements = random.randint(3, 4)
    if removable_neighbours.size < num_replacements:
        num_replacements = removable_neighbours.size
    removed_neighbours = np.random.choice(removable_neighbours, num_replacements, replace=False)
    selection_mask[removed_neighbours] = False
    covered_rows = (coverage_matrix @ selection_mask) > 0

    #add valid round trips until all legs are covered
    while not np.array_equal(covered_rows, required_rows):
        available_round_trips = np.where(np.all(coverage_matrix[covered_rows, :] == 0, axis=0))[0]
        if available_round_trips.size == 0:
            #if stuck, restart neighbor generation from the current solution
            return generate_neighbor_solution(current_solution_indices, coverage_matrix)
        chosen_round_trip = np.random.choice(available_round_trips)
        selection_mask[chosen_round_trip] = True
        covered_rows = (coverage_matrix @ selection_mask) > 0

    selected_indices = np.where(selection_mask)[0]
    return selected_indices

def simulated_annealing(rows, cols, costs, coverage):
    #convert coverage (list of sets) to a NumPy matrix for neighbor generation
    coverage_matrix = convert_coverage_to_matrix(coverage, rows, cols)
    
    #generate initial solution
    current_solution_indices = initialise_valid_solution(coverage_matrix)
    current_solution = convert_indices_to_binary(current_solution_indices, cols)
    current_solution_cost = sum(costs[i] for i in range(cols) if current_solution[i] == 1)
    
    #simulated Annealing parameters 
    initial_temperature = 4000
    cooling_decrease = 1
    temperature = initial_temperature
    num_iterations = 10000  # Number of iterations
    
    for iteration in range(num_iterations):
        #generate neighbor solution (in indices form)
        neighbor_solution_indices = generate_neighbor_solution(current_solution_indices, coverage_matrix)
        neighbor_solution = convert_indices_to_binary(neighbor_solution_indices, cols)
        neighbor_solution_cost = sum(costs[i] for i in range(cols) if neighbor_solution[i] == 1)
        
        #calculate the diff between the current cost and the neighbour cost and decide on acceptance using the Metropolis criterion
        delta = neighbor_solution_cost - current_solution_cost
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_solution_indices = neighbor_solution_indices
            current_solution = neighbor_solution
            current_solution_cost = neighbor_solution_cost
        
        #update temperature (linear cooling)
        temperature -= cooling_decrease
        if temperature < 0.01:
            temperature = 0.01

    return current_solution, current_solution_cost


def is_feasible(solution, coverage, rows):
    """Check if a solution covers each row exactly once."""
    covered = Counter()
    for i in range(len(solution)):
        if solution[i] == 1:
            for row in coverage[i]:
                covered[row] += 1
    return all(covered[row] == 1 for row in range(1, rows + 1))

def print_solution(solution, costs, coverage, rows):
    """Prints the selected schedules and total cost."""
    selected_schedules = [(schedule_id, costs[schedule_id], coverage[schedule_id]) 
                          for schedule_id in range(len(solution)) if solution[schedule_id] == 1]

    print("\n==== Simulated Annealing Solution ====")
    print(f"Total Selected Crew Schedules: {len(selected_schedules)}")

    if not selected_schedules:
        print("No schedules selected")
        return False, 0
    
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
    
    # Verify feasibility
    feasible = is_feasible(solution, coverage, rows)
    print("Solution is feasible:", feasible)
    
    return feasible, total_cost

def evaluate_algorithm(file_path, num_trials=30):
    """Evaluates the simulated annealing algorithm across multiple trials."""
    successful_runs = 0
    total_costs = []
    execution_times = []
    
    rows, cols, costs, coverage = parse_problem(file_path)
    
    for trial in range(num_trials):
        print(f"Trial {trial+1}/{num_trials}...")
        start_time = time.time()
        solution, cost = simulated_annealing(rows, cols, costs, coverage)
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
file_path = 'datasets/sppnw41.txt'
rows, cols, costs, coverage = parse_problem(file_path)
print(f"Running simulated annealing on {file_path}")
best_solution, best_cost = simulated_annealing(rows, cols, costs, coverage)
print_solution(best_solution, costs, coverage, rows)

#evaluate 
#uncooment this to evaluate the algorithm over 30 runs
# benchmark_files = ["datasets/sppnw41.txt", "datasets/sppnw42.txt", "datasets/sppnw43.txt"]
# benchmark_results = {file: evaluate_algorithm(file) for file in benchmark_files}
# df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
# print(df_results)
