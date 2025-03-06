import random
import math
import time
import numpy as np
import pandas as pd
from collections import Counter

# --- Data Import and Conversion Functions ---
def parse_problem(file_path):
    """
    Parse problem data using a NumPy-based approach (as in dave.py) and then convert
    the coverage matrix into a list of sets (to keep your is_feasible and print_solution functions unchanged).
    
    Returns:
        rows: number of flight legs.
        cols: number of round trips (crew schedules).
        costs: list of costs for each round trip.
        coverage: list of sets where each set contains the flight legs (1-indexed) covered by that round trip.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # First line: number of flight legs (rows) and round trips (cols)
        rows, cols = map(int, lines[0].split())
        
        # Initialize NumPy arrays for costs and a binary coverage matrix
        costs_array = np.zeros(cols, dtype=int)
        coverage_matrix = np.zeros((rows, cols), dtype=int)
        line_index = 1
        for j in range(cols):
            column_data = list(map(int, lines[line_index].split()))
            costs_array[j] = column_data[0]
            legs = column_data[2:]
            for leg in legs:
                # Note: legs in file are 1-indexed; matrix rows are 0-indexed
                coverage_matrix[leg - 1, j] = 1
            line_index += 1
        
        # Convert coverage_matrix to list-of-sets for compatibility with is_feasible and print_solution
        coverage = []
        for j in range(cols):
            col_set = {i + 1 for i in range(rows) if coverage_matrix[i, j] == 1}
            coverage.append(col_set)
        costs = costs_array.tolist()
        return rows, cols, costs, coverage
    except Exception as e:
        print(f"Error parsing problem: {e}")
        return None, None, None, None

def convert_coverage_to_matrix(coverage, rows, cols):
    """
    Convert a list-of-sets (coverage) to a NumPy binary matrix.
    Each column j of the returned matrix has a 1 in row i if (i+1) is in coverage[j].
    """
    mat = np.zeros((rows, cols), dtype=int)
    for j in range(cols):
        for r in coverage[j]:
            mat[r - 1, j] = 1
    return mat

def convert_indices_to_binary(selected_indices, cols):
    """
    Given an array of selected round trip indices, return a binary vector (list) of length cols.
    """
    sol = [0] * cols
    for i in selected_indices:
        sol[i] = 1
    return sol

# --- dave.py–Style Initial Solution and Neighbor Generation ---
def initialise_valid_solution(coverage_matrix):
    """
    Generates a random valid initial solution ensuring all flight legs are covered exactly once.
    Uses the dave.py logic with NumPy.
    
    Args:
        coverage_matrix: A NumPy binary matrix (rows x cols).
    
    Returns:
        A NumPy array of selected round trip indices.
    """
    n_legs, n_round_trips = coverage_matrix.shape
    selected_round_trips = np.zeros(n_round_trips, dtype=bool)
    covered_legs = np.zeros(n_legs, dtype=bool)
    required_legs = np.ones(n_legs, dtype=bool)

    # Step 1: Select a random round trip to start
    first_round_trip = random.randint(0, n_round_trips - 1)
    selected_round_trips[first_round_trip] = True
    covered_legs = (coverage_matrix @ selected_round_trips) > 0

    # Step 2: Add more round trips until all legs are covered
    while not np.array_equal(covered_legs, required_legs):
        available_round_trips = np.where(np.all(coverage_matrix[covered_legs, :] == 0, axis=0))[0]
        if available_round_trips.size == 0:
            # Restart if no valid replacement is found
            return initialise_valid_solution(coverage_matrix)
        chosen_round_trip = np.random.choice(available_round_trips)
        selected_round_trips[chosen_round_trip] = True
        covered_legs = (coverage_matrix @ selected_round_trips) > 0

    selected_indices = np.where(selected_round_trips)[0]
    # Uncomment the next line to print the initial solution (if desired)
    # print("Initial valid solution indices:", selected_indices)
    return selected_indices

def generate_neighbor_solution(current_solution_indices, coverage_matrix):
    """
    Generates a neighboring solution by removing 3-4 round trips at random and then
    adding valid replacements until all legs are covered.
    
    Args:
        current_solution_indices: A NumPy array of selected round trip indices.
        coverage_matrix: A NumPy binary matrix.
    
    Returns:
        A new NumPy array of selected round trip indices.
    """
    n_legs, n_round_trips = coverage_matrix.shape
    # Convert current solution indices into a boolean selection mask
    selection_mask = np.zeros(n_round_trips, dtype=bool)
    selection_mask[current_solution_indices] = True
    covered_legs = (coverage_matrix @ selection_mask) > 0
    required_legs = np.ones(n_legs, dtype=bool)

    # Remove 3-4 round trips at random
    removable_trips = np.where(selection_mask)[0]
    num_removals = random.randint(3, 4)
    if removable_trips.size < num_removals:
        num_removals = removable_trips.size
    removed_trips = np.random.choice(removable_trips, num_removals, replace=False)
    selection_mask[removed_trips] = False
    covered_legs = (coverage_matrix @ selection_mask) > 0

    # Add valid round trips until all legs are covered
    while not np.array_equal(covered_legs, required_legs):
        available_round_trips = np.where(np.all(coverage_matrix[covered_legs, :] == 0, axis=0))[0]
        if available_round_trips.size == 0:
            # If stuck, restart neighbor generation from the current solution
            return generate_neighbor_solution(current_solution_indices, coverage_matrix)
        chosen_round_trip = np.random.choice(available_round_trips)
        selection_mask[chosen_round_trip] = True
        covered_legs = (coverage_matrix @ selection_mask) > 0

    selected_indices = np.where(selection_mask)[0]
    return selected_indices

# --- Simulated Annealing (using dave.py logic) ---
def simulated_annealing(rows, cols, costs, coverage):
    """
    Simplified simulated annealing using dave.py logic.
    Uses a linear cooling schedule and the Metropolis criterion.
    
    Args:
        rows: Number of flight legs.
        cols: Number of round trips.
        costs: List of costs for each round trip.
        coverage: List of sets (each set contains flight legs covered by that round trip).
        
    Returns:
        A tuple (solution, cost) where solution is a binary list of length cols.
    """
    # Convert coverage (list of sets) to a NumPy matrix for neighbor generation
    coverage_matrix = convert_coverage_to_matrix(coverage, rows, cols)
    
    # Generate initial solution using dave.py logic
    current_solution_indices = initialise_valid_solution(coverage_matrix)
    current_solution = convert_indices_to_binary(current_solution_indices, cols)
    current_solution_cost = sum(costs[i] for i in range(cols) if current_solution[i] == 1)
    
    # Simulated Annealing parameters (as in dave.py)
    initial_temperature = 4000
    cooling_decrease = 1
    T = initial_temperature
    k = 10000  # Number of iterations
    
    for t in range(k):
        # Generate neighbor solution (in indices form)
        neighbor_solution_indices = generate_neighbor_solution(current_solution_indices, coverage_matrix)
        neighbor_solution = convert_indices_to_binary(neighbor_solution_indices, cols)
        neighbor_solution_cost = sum(costs[i] for i in range(cols) if neighbor_solution[i] == 1)
        
        # Calculate energy difference and decide on acceptance using the Metropolis criterion
        delta_E = neighbor_solution_cost - current_solution_cost
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            current_solution_indices = neighbor_solution_indices
            current_solution = neighbor_solution
            current_solution_cost = neighbor_solution_cost
        
        # Update temperature (linear cooling)
        T -= cooling_decrease
        if T < 0.01:
            T = 0.01

    return current_solution, current_solution_cost

# --- Your Existing Functions (kept unchanged) ---
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

# --- Main Block ---
if __name__ == "__main__":
    # Example: run on a single dataset
    file_path = 'datasets/sppnw41.txt'
    rows, cols, costs, coverage = parse_problem(file_path)
    print(f"Running simulated annealing on {file_path}")
    best_solution, best_cost = simulated_annealing(rows, cols, costs, coverage)
    print_solution(best_solution, costs, coverage, rows)

    # Uncomment the following block to evaluate on multiple benchmark files:
    benchmark_files = ["datasets/sppnw41.txt", "datasets/sppnw42.txt", "datasets/sppnw43.txt"]
    benchmark_results = {file: evaluate_algorithm(file) for file in benchmark_files}
    df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
    print(df_results)
