import random
import math
import time
import numpy as np
import pandas as pd
from collections import Counter

def parse_problem(file_path):
    """Parse problem data from file."""
    with open(file_path, 'r') as f:
        rows, cols = map(int, f.readline().split())
        costs, coverage = [], []
        for _ in range(cols):
            parts = list(map(int, f.readline().split()))
            cost = parts[0]
            costs.append(cost)
            num_flights = parts[1]
            flights = set(parts[2 : 2 + num_flights])
            coverage.append(flights)
        return rows, cols, costs, coverage

def is_feasible(solution, coverage, rows):
    """Check if a solution covers each row exactly once."""
    covered = Counter()
    for i in range(len(solution)):
        if solution[i] == 1:
            for row in coverage[i]:
                covered[row] += 1
    
    return all(covered[row] == 1 for row in range(1, rows + 1))

def calculate_cost(solution, costs, coverage, rows, alpha=10000, beta=5000):
    """Calculate cost with penalties for infeasible solutions."""
    # Base cost from selected columns
    base_cost = sum(costs[i] for i in range(len(solution)) if solution[i] == 1)
    
    # Calculate coverage statistics
    covered = Counter()
    for i in range(len(solution)):
        if solution[i] == 1:
            for row in coverage[i]:
                covered[row] += 1
    
    # Count constraint violations
    uncovered = set(range(1, rows + 1)) - covered.keys()
    uncovered_penalty = len(uncovered) * alpha
    
    # Penalize overcovered rows
    overcovered_penalty = sum((count - 1) for count in covered.values() if count > 1) * beta
    
    total_cost = base_cost + uncovered_penalty + overcovered_penalty
    return total_cost, base_cost, len(uncovered), sum(1 for count in covered.values() if count > 1)

def greedy_construction(rows, cols, costs, coverage):
    """Build a solution using a greedy construction heuristic."""
    solution = [0] * cols
    uncovered = set(range(1, rows + 1))
    
    while uncovered:
        best_ratio = float('inf')
        best_col = -1
        
        for i in range(cols):
            if solution[i] == 0:  # Only consider columns not yet in solution
                new_coverage = uncovered.intersection(coverage[i])
                if new_coverage:
                    # Calculate efficiency ratio (cost per newly covered row)
                    ratio = costs[i] / len(new_coverage)
                    if ratio < best_ratio:
                        best_ratio = ratio
                        best_col = i
        
        if best_col == -1:
            # No column can cover more uncovered rows, add a random one
            candidates = [i for i in range(cols) if solution[i] == 0 and 
                        any(row in coverage[i] for row in uncovered)]
            if candidates:
                best_col = random.choice(candidates)
            else:
                break  # No more useful columns found
        
        # Add the chosen column
        if best_col != -1:
            solution[best_col] = 1
            uncovered -= coverage[best_col]
    
    return solution

def generate_neighbor(solution, coverage, rows, temp, current_temp):
    """Generate a neighboring solution with adaptive strategy selection."""
    neighbor = solution.copy()
    
    # Calculate current coverage
    covered = Counter()
    for i in range(len(solution)):
        if solution[i] == 1:
            for row in coverage[i]:
                covered[row] += 1
    
    # Identify uncovered and overcovered rows
    uncovered = set(range(1, rows + 1)) - covered.keys()
    overcovered = {row for row, count in covered.items() if count > 1}
    
    # Determine move type based on current state and temperature
    temp_ratio = current_temp / temp  # Normalized temperature (1.0 to ~0)
    
    if uncovered:
        # If there are uncovered rows, favor adding columns
        move_type = 'add'
    elif overcovered and random.random() < 0.7:
        # If there are overcovered rows, try to remove columns
        move_type = 'remove'
    else:
        # Exploration vs. exploitation based on temperature
        if temp_ratio > 0.3:  # Higher temperature, more exploration
            move_type = random.choice(['flip', 'swap', 'cluster'])
        else:  # Lower temperature, more focused moves
            move_type = random.choice(['swap', 'remove', 'cluster'])
    
    # Apply the selected move
    if move_type == 'flip':
        # Simple bit flip
        idx = random.randint(0, len(solution) - 1)
        neighbor[idx] = 1 - neighbor[idx]
        
    elif move_type == 'swap':
        # Swap a 0 and a 1
        ones = [i for i in range(len(solution)) if solution[i] == 1]
        zeros = [i for i in range(len(solution)) if solution[i] == 0]
        
        if ones and zeros:
            idx1 = random.choice(ones)
            idx2 = random.choice(zeros)
            neighbor[idx1] = 0
            neighbor[idx2] = 1
    
    elif move_type == 'add':
        # Add a column that might cover uncovered rows
        zeros = [i for i in range(len(solution)) if solution[i] == 0]
        
        if uncovered and zeros:
            # Prioritize columns that cover uncovered rows
            candidates = [i for i in zeros if any(row in coverage[i] for row in uncovered)]
            if candidates:
                idx = random.choice(candidates)
            else:
                idx = random.choice(zeros)
            neighbor[idx] = 1
    
    elif move_type == 'remove':
        # Remove a column, preferably one causing overcoverage
        ones = [i for i in range(len(solution)) if solution[i] == 1]
        
        if ones and len(ones) > 1:  # Ensure we keep at least one column
            if overcovered:
                # Prefer columns that contribute to overcoverage
                candidates = [i for i in ones if any(row in coverage[i] for row in overcovered)]
                if candidates:
                    idx = random.choice(candidates)
                else:
                    idx = random.choice(ones)
            else:
                idx = random.choice(ones)
            neighbor[idx] = 0
    
    elif move_type == 'cluster':
        # Make multiple related changes focused on a specific row
        row = random.randint(1, rows)
        
        # Find columns that cover this row
        covering_cols = [i for i in range(len(solution)) if row in coverage[i]]
        
        if covering_cols:
            current_covering = [i for i in covering_cols if solution[i] == 1]
            potential_covering = [i for i in covering_cols if solution[i] == 0]
            
            if current_covering and potential_covering:
                # Replace one current column with a potential column
                to_remove = random.choice(current_covering)
                to_add = random.choice(potential_covering)
                neighbor[to_remove] = 0
                neighbor[to_add] = 1
    
    return neighbor

def repair_solution(solution, costs, coverage, rows):
    """Repair an infeasible solution to ensure feasibility."""
    repaired = solution.copy()
    
    # First check which rows are currently covered
    covered = set()
    for i in range(len(repaired)):
        if repaired[i] == 1:
            covered.update(coverage[i])
    
    # Add columns to cover uncovered rows
    uncovered = set(range(1, rows + 1)) - covered
    while uncovered:
        best_col = -1
        best_efficiency = -1
        
        for i in range(len(repaired)):
            if repaired[i] == 0:  # Only consider unused columns
                new_coverage = uncovered.intersection(coverage[i])
                if new_coverage:
                    efficiency = len(new_coverage) / costs[i]
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_col = i
        
        if best_col == -1:
            # Fallback to any column that covers at least one uncovered row
            candidates = [i for i in range(len(repaired)) if repaired[i] == 0 and 
                        any(row in coverage[i] for row in uncovered)]
            if candidates:
                best_col = random.choice(candidates)
            else:
                # Add any column at this point
                for i in range(len(repaired)):
                    if repaired[i] == 0:
                        for row in uncovered:
                            if row in coverage[i]:
                                repaired[i] = 1
                                covered.update(coverage[i])
                                break
                
                # Recalculate uncovered rows
                uncovered = set(range(1, rows + 1)) - covered
                continue
        
        # Add the best column
        repaired[best_col] = 1
        covered.update(coverage[best_col])
        uncovered = set(range(1, rows + 1)) - covered
    
    # Remove redundant columns (if removing doesn't make solution infeasible)
    # Sort columns by cost (descending) to try removing expensive ones first
    columns_to_check = [(i, costs[i]) for i in range(len(repaired)) if repaired[i] == 1]
    columns_to_check.sort(key=lambda x: x[1], reverse=True)
    
    for col_idx, _ in columns_to_check:
        # Temporarily remove this column
        repaired[col_idx] = 0
        
        # Check if solution is still feasible
        if not is_feasible(repaired, coverage, rows):
            # If not feasible, put it back
            repaired[col_idx] = 1
    
    return repaired

def enhanced_simulated_annealing(rows, cols, costs, coverage, 
                               max_iterations=250000,
                               initial_temp=5000,
                               min_temp=1e-6,
                               no_improve_limit=15000,
                               repair_frequency=0.15,
                               alpha=100000,  # Penalty for uncovered rows
                               beta=20000):   # Penalty for overcovered rows
    """Enhanced simulated annealing algorithm with adaptive strategies."""
    # Start with a greedy initial solution
    current_solution = greedy_construction(rows, cols, costs, coverage)
    
    # Calculate initial cost
    current_total_cost, current_base_cost, uncovered, overcovered = calculate_cost(
        current_solution, costs, coverage, rows, alpha, beta)
    
    # Track best solution
    best_solution = current_solution.copy()
    best_cost = current_base_cost if is_feasible(current_solution, coverage, rows) else float('inf')
    
    # Initialize tracking variables
    temperature = initial_temp
    iteration = 0
    no_improve_counter = 0
    best_feasible_found = is_feasible(current_solution, coverage, rows)
    
    # For adaptive penalty weights
    current_alpha = alpha
    current_beta = beta
    
    while iteration < max_iterations and temperature > min_temp and no_improve_counter < no_improve_limit:
        iteration += 1
        
        # Periodically adjust penalty weights based on solution status
        if iteration % 1000 == 0:
            if uncovered > 0:
                current_alpha *= 1.05
            elif uncovered == 0:
                current_alpha = max(alpha, current_alpha * 0.95)
                
            if overcovered > 0:
                current_beta *= 1.05
            elif overcovered == 0:
                current_beta = max(beta, current_beta * 0.95)
        
        # Generate neighbor with adaptive strategy
        neighbor = generate_neighbor(current_solution, coverage, rows, initial_temp, temperature)
        
        # Periodically repair solution to ensure feasibility exploration
        if random.random() < repair_frequency:
            neighbor = repair_solution(neighbor, costs, coverage, rows)
        
        # Calculate new solution cost
        new_total_cost, new_base_cost, new_uncovered, new_overcovered = calculate_cost(
            neighbor, costs, coverage, rows, current_alpha, current_beta)
        
        # Determine whether to accept the new solution
        accept = False
        
        if new_total_cost < current_total_cost:
            # Always accept better solutions
            accept = True
        else:
            # For worse solutions, use temperature to determine acceptance
            delta = new_total_cost - current_total_cost
            # Adaptive acceptance probability based on temperature
            prob = math.exp(-delta / temperature)
            accept = random.random() < prob
        
        if accept:
            # Update current solution
            current_solution = neighbor
            current_total_cost = new_total_cost
            current_base_cost = new_base_cost
            uncovered = new_uncovered
            overcovered = new_overcovered
            
            # Check if this is a new best feasible solution
            if is_feasible(current_solution, coverage, rows) and current_base_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_base_cost
                best_feasible_found = True
                no_improve_counter = 0
            else:
                no_improve_counter += 1
        else:
            no_improve_counter += 1
        
        # Update temperature - adaptive cooling
        progress = iteration / max_iterations
        if progress < 0.3:  # Early exploration phase
            temperature = initial_temp * (1 - 0.3 * progress)
        elif progress < 0.7:  # Middle intensification phase
            temperature = initial_temp * 0.7 * (1 - (progress - 0.3) / 0.4)
        else:  # Final refinement phase
            temperature = initial_temp * 0.1 * (1 - (progress - 0.7) / 0.3)
        
        # Output progress periodically
        if iteration % 10000 == 0:
            print(f"Iteration {iteration}, Temp: {temperature:.4f}, Current cost: {current_base_cost}, " 
                  f"Best: {best_cost if best_feasible_found else 'None'}, "
                  f"Uncovered: {uncovered}, Overcovered: {overcovered}")
    
    # If the best solution found is not feasible, repair it
    if not best_feasible_found or not is_feasible(best_solution, coverage, rows):
        repaired_solution = repair_solution(current_solution, costs, coverage, rows)
        repaired_cost = sum(costs[i] for i in range(len(repaired_solution)) if repaired_solution[i] == 1)
        
        if is_feasible(repaired_solution, coverage, rows) and (not best_feasible_found or repaired_cost < best_cost):
            best_solution = repaired_solution
            best_cost = repaired_cost
    
    return best_solution, best_cost

def simulated_annealing_with_restarts(rows, cols, costs, coverage, num_restarts=5, **params):
    """Run simulated annealing multiple times and return the best solution."""
    best_solution = None
    best_cost = float('inf')
    
    for restart in range(num_restarts):
        print(f"Restart {restart+1}/{num_restarts}")
        solution, cost = enhanced_simulated_annealing(rows, cols, costs, coverage, **params)
        
        if is_feasible(solution, coverage, rows) and cost < best_cost:
            best_solution = solution.copy()
            best_cost = cost
            
            # Early stopping if we find a solution that's very close to the optimal
            if dataset_optimal_costs.get(current_dataset, 0) > 0:
                gap = (cost - dataset_optimal_costs[current_dataset]) / dataset_optimal_costs[current_dataset]
                if gap < 0.05:  # Stop if within 5% of optimal
                    print(f"Found near-optimal solution (gap: {gap:.2%}), stopping early.")
                    break
    
    return best_solution, best_cost

def print_solution(solution, costs, coverage, rows):
    """Print solution details and verify feasibility."""
    selected_schedules = [(i, costs[i], coverage[i]) for i in range(len(solution)) if solution[i] == 1]
    print("\n==== Simulated Annealing Solution ====")
    print(f"Total Selected Crew Schedules: {len(selected_schedules)}")
    if not selected_schedules:
        print("No schedules selected")
        return
    total_cost = sum(cost for _, cost, _ in selected_schedules)
    print(f"Total Cost: {total_cost}")
    print("\nSelected Crew Schedules:")
    for idx, cost, flights in selected_schedules[:10]:  # Print first 10 for brevity
        print(f"- Schedule {idx}: Cost = {cost}, Covers Flights: {sorted(flights)}")
    if len(selected_schedules) > 10:
        print(f"... and {len(selected_schedules) - 10} more schedules (omitted for brevity)")
    
    # Verify the solution coverage
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1
    
    # Count coverage issues
    uncovered = sum(1 for count in flight_coverage if count == 0)
    overcovered = sum(1 for count in flight_coverage if count > 1)
    correctly_covered = sum(1 for count in flight_coverage if count == 1)
    
    print("====================================")
    print(f"Coverage Statistics:")
    print(f"- Correctly covered flights: {correctly_covered}/{rows} ({correctly_covered/rows*100:.2f}%)")
    print(f"- Uncovered flights: {uncovered}")
    print(f"- Overcovered flights: {overcovered}")
    print("====================================\n")
    
    # Verify feasibility
    feasible = is_feasible(solution, coverage, rows)
    print("Solution is feasible:", feasible)
    return feasible, total_cost

def evaluate_algorithm(file_path, num_trials=5, **kwargs):
    """Evaluate the algorithm on a specific problem instance."""
    global current_dataset
    current_dataset = file_path
    
    rows, cols, costs, coverage = parse_problem(file_path)
    
    successful_runs = 0
    total_costs = []
    execution_times = []
    
    for trial in range(num_trials):
        print(f"Trial {trial+1}/{num_trials} for {file_path}...")
        start_time = time.time()
        solution, cost = simulated_annealing_with_restarts(rows, cols, costs, coverage, **kwargs)
        feasible = is_feasible(solution, coverage, rows)
        duration = time.time() - start_time
        
        if feasible:
            successful_runs += 1
            total_costs.append(cost)
        else:
            print(f"Warning: Trial {trial+1} produced an infeasible solution!")
        
        execution_times.append(duration)
    
    success_rate = (successful_runs / num_trials) * 100
    average_cost = np.mean(total_costs) if total_costs else float('inf')
    min_cost = min(total_costs) if total_costs else float('inf')
    std_dev_cost = np.std(total_costs) if total_costs else float('inf')
    average_time = np.mean(execution_times)
    
    results = {
        "Success Rate (%)": success_rate,
        "Average Cost": average_cost,
        "Minimum Cost": min_cost,
        "Standard Deviation": std_dev_cost,
        "Average Execution Time (s)": average_time
    }
    
    print(f"Results for {file_path}:")
    for key, value in results.items():
        print(f"- {key}: {value}")
    print()
    
    return results

# Optimal known costs for each dataset
dataset_optimal_costs = {
    "datasets/sppnw41.txt": 10972.5,
    "datasets/sppnw42.txt": 7485.0,
    "datasets/sppnw43.txt": 8897.0
}

# Global variable to track current dataset
current_dataset = ""

# Tuned parameters for each dataset
dataset_parameters = {
    "datasets/sppnw41.txt": {
        "max_iterations": 150000,
        "initial_temp": 5000,
        "min_temp": 1e-6,
        "no_improve_limit": 10000,
        "repair_frequency": 0.2,
        "alpha": 100000,
        "beta": 20000,
        "num_restarts": 3
    },
    "datasets/sppnw42.txt": {
        "max_iterations": 200000,
        "initial_temp": 8000,
        "min_temp": 1e-6,
        "no_improve_limit": 20000,
        "repair_frequency": 0.25,
        "alpha": 150000,
        "beta": 30000,
        "num_restarts": 5
    },
    "datasets/sppnw43.txt": {
        "max_iterations": 180000,
        "initial_temp": 6000,
        "min_temp": 1e-6,
        "no_improve_limit": 15000,
        "repair_frequency": 0.2,
        "alpha": 120000,
        "beta": 25000,
        "num_restarts": 4
    }
}

if __name__ == "__main__":
    # Test on all datasets
    benchmark_files = ["datasets/sppnw41.txt", "datasets/sppnw42.txt", "datasets/sppnw43.txt"]
    all_results = {}
    
    for file_path in benchmark_files:
        print(f"\n{'='*30}\nProcessing {file_path}\n{'='*30}")
        rows, cols, costs, coverage = parse_problem(file_path)
        
        # Get parameters for this dataset
        params = dataset_parameters[file_path]
        
        # Run main algorithm
        solution, cost = simulated_annealing_with_restarts(rows, cols, costs, coverage, **params)
        feasible, actual_cost = print_solution(solution, costs, coverage, rows)
        
        print(f"Found solution cost: {cost}")
        print(f"Optimal cost for this dataset: {dataset_optimal_costs[file_path]}")
        gap = ((cost - dataset_optimal_costs[file_path]) / dataset_optimal_costs[file_path]) * 100
        print(f"Gap to optimal: {gap:.4f}%")
        
        # Full evaluation with fewer trials for demonstration
        all_results[file_path] = evaluate_algorithm(file_path, num_trials=2, **params)
    
    # Print summary
    print("\n===== SUMMARY OF RESULTS =====")
    summary_df = pd.DataFrame.from_dict(all_results, orient='index')
    print(summary_df)
    print("\nNote: For final results, increase num_trials to 10 or more.")