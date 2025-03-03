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

def get_coverage_stats(solution, coverage, rows):
    """Calculate coverage statistics for a solution."""
    covered = Counter()
    for i in range(len(solution)):
        if solution[i] == 1:
            for row in coverage[i]:
                covered[row] += 1
    
    # Count uncovered and over-covered rows
    uncovered = rows - len(covered)
    overcovered = sum(1 for count in covered.values() if count > 1)
    
    return {
        'covered_exactly_once': sum(1 for count in covered.values() if count == 1),
        'uncovered': uncovered,
        'overcovered': overcovered
    }

def calculate_cost(solution, costs, coverage, rows, alpha=130000, beta=25000):
    """
    Calculate cost with adaptive penalty weights for infeasible solutions.
    - alpha: penalty weight for uncovered rows
    - beta: penalty weight for overcovered rows
    """
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

def is_feasible(solution, coverage, rows):
    """Check if a solution covers each row exactly once."""
    covered = Counter()
    for i in range(len(solution)):
        if solution[i] == 1:
            for row in coverage[i]:
                covered[row] += 1
    
    return all(covered[row] == 1 for row in range(1, rows + 1))

def greedy_construction(rows, cols, costs, coverage):
    """
    Build a solution using a greedy construction heuristic.
    Prioritizes cost-effective columns that cover uncovered rows.
    """
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

def random_adaptive_solution(rows, cols, costs, coverage, greedy_prob=0.7):
    """
    Generate an initial solution with a mix of greedy and random strategies.
    Has a probability of using the greedy approach or a randomized approach.
    """
    if random.random() < greedy_prob:
        # Use greedy construction most of the time
        return greedy_construction(rows, cols, costs, coverage)
    else:
        # Sometimes use a randomized approach
        solution = [0] * cols
        uncovered = set(range(1, rows + 1))
        
        # Shuffle columns to add randomness
        column_indices = list(range(cols))
        random.shuffle(column_indices)
        
        for i in column_indices:
            if uncovered.intersection(coverage[i]):
                solution[i] = 1
                uncovered -= coverage[i]
            
            if not uncovered:
                break
        
        return solution

def generate_neighbor(solution, coverage, rows, move_strategy='adaptive'):
    """
    Generate a neighboring solution using various move strategies.
    
    Parameters:
    - solution: Current solution
    - coverage: List of sets indicating which rows each column covers
    - rows: Total number of rows to cover
    - move_strategy: Strategy for generating neighbors
      ('flip', 'swap', 'adaptive', 'tabu')
    
    Returns:
    - neighbor: The generated neighboring solution
    """
    neighbor = solution.copy()
    
    # Determine move type adaptively
    if move_strategy == 'adaptive':
        # Calculate current coverage status
        stats = get_coverage_stats(solution, coverage, rows)
        
        if stats['uncovered'] > 0:
            # If there are uncovered rows, favor adding columns
            move_strategy = 'add'
        elif stats['overcovered'] > 0 and random.random() < 0.7:
            # If there are overcovered rows, try to remove columns
            move_strategy = 'remove'
        else:
            # Otherwise, choose randomly
            move_strategy = random.choice(['flip', 'swap', 'cluster'])
    
    # Apply the selected move
    if move_strategy == 'flip':
        # Simple bit flip
        idx = random.randint(0, len(solution) - 1)
        neighbor[idx] = 1 - neighbor[idx]
        
    elif move_strategy == 'swap':
        # Swap a 0 and a 1
        ones = [i for i in range(len(solution)) if solution[i] == 1]
        zeros = [i for i in range(len(solution)) if solution[i] == 0]
        
        if ones and zeros:
            idx1 = random.choice(ones)
            idx2 = random.choice(zeros)
            neighbor[idx1] = 0
            neighbor[idx2] = 1
    
    elif move_strategy == 'add':
        # Add a column that might cover uncovered rows
        zeros = [i for i in range(len(solution)) if solution[i] == 0]
        if zeros:
            idx = random.choice(zeros)
            neighbor[idx] = 1
    
    elif move_strategy == 'remove':
        # Remove a column if possible
        ones = [i for i in range(len(solution)) if solution[i] == 1]
        if ones and len(ones) > 1:  # Ensure we keep at least one column
            idx = random.choice(ones)
            neighbor[idx] = 0
    
    elif move_strategy == 'cluster':
        # Make multiple related changes
        # First, randomly select a row to focus on
        row = random.randint(1, rows)
        
        # Find columns that cover this row
        covering_cols = [i for i in range(len(solution)) if row in coverage[i]]
        
        if covering_cols:
            # If the row is currently covered, try to change which column covers it
            current_covering = [i for i in covering_cols if solution[i] == 1]
            potential_covering = [i for i in covering_cols if solution[i] == 0]
            
            if current_covering and potential_covering:
                # Remove one current column and add one potential column
                to_remove = random.choice(current_covering)
                to_add = random.choice(potential_covering)
                neighbor[to_remove] = 0
                neighbor[to_add] = 1
    
    return neighbor

def repair_solution(solution, costs, coverage, rows):
    """
    Repair an infeasible solution by ensuring all rows are covered
    while trying to minimize cost.
    """
    repaired = solution.copy()
    
    # Check which rows are currently covered
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
            # No column can cover more rows, try a random one
            candidates = [i for i in range(len(repaired)) if repaired[i] == 0 and 
                        any(row in coverage[i] for row in uncovered)]
            if candidates:
                best_col = random.choice(candidates)
            else:
                # This is an extreme case - add columns that cover any uncovered row
                for i in range(len(repaired)):
                    if repaired[i] == 0:
                        for row in uncovered:
                            if row in coverage[i]:
                                repaired[i] = 1
                                covered.update(coverage[i])
                                break
                
                # Recalculate uncovered
                uncovered = set(range(1, rows + 1)) - covered
                continue
        
        # Add the best column
        repaired[best_col] = 1
        covered.update(coverage[best_col])
        uncovered = set(range(1, rows + 1)) - covered
    
    # Try to remove redundant columns
    for i in range(len(repaired)):
        if repaired[i] == 1:
            # Temporarily remove this column
            repaired[i] = 0
            
            # Check if all rows are still covered
            temp_covered = set()
            for j in range(len(repaired)):
                if repaired[j] == 1:
                    temp_covered.update(coverage[j])
            
            # If removing makes solution infeasible, put it back
            if len(temp_covered) < rows:
                repaired[i] = 1
    
    return repaired

def adaptive_cooling_schedule(initial_temp, iteration, max_iterations, method='adaptive'):
    """
    Calculate current temperature using various cooling schedules.
    
    Parameters:
    - initial_temp: Initial temperature
    - iteration: Current iteration
    - max_iterations: Maximum number of iterations
    - method: Cooling method ('geometric', 'linear', 'slow_start', 'slow_end', 'adaptive')
    
    Returns:
    - Current temperature
    """
    if method == 'geometric':
        # Standard geometric cooling
        alpha = 0.99  # Cooling rate
        return initial_temp * (alpha ** iteration)
    
    elif method == 'linear':
        # Linear cooling
        return initial_temp * (1 - iteration / max_iterations)
    
    elif method == 'slow_start':
        # Slow at the beginning, faster at the end
        return initial_temp * (1 - (iteration / max_iterations) ** 2)
    
    elif method == 'slow_end':
        # Fast at the beginning, slower at the end
        return initial_temp * (1 - math.sqrt(iteration / max_iterations))
    
    elif method == 'adaptive':
        # Adaptive cooling that adjusts based on progress
        # Starts slow, becomes aggressive in the middle, then slows down at the end
        progress = iteration / max_iterations
        
        if progress < 0.3:  # Early exploration phase
            return initial_temp * (1 - 0.3 * progress)
        elif progress < 0.7:  # Middle intensification phase
            return initial_temp * 0.7 * (1 - (progress - 0.3) / 0.4)
        else:  # Final refinement phase
            return initial_temp * 0.1 * (1 - (progress - 0.7) / 0.3)

def acceptance_probability(current_cost, new_cost, temperature, method='boltzmann'):
    """
    Calculate the probability of accepting a worse solution.
    
    Parameters:
    - current_cost: Cost of current solution
    - new_cost: Cost of new solution
    - temperature: Current temperature
    - method: Method for calculating probability
    
    Returns:
    - Probability of accepting the new solution (0-1)
    """
    if new_cost <= current_cost:
        return 1.0  # Always accept better solutions
    
    delta = new_cost - current_cost
    
    if method == 'boltzmann':
        # Standard Boltzmann distribution
        return math.exp(-delta / temperature)
    
    elif method == 'modified':
        # Modified function that's more selective at higher costs
        return math.exp(-delta / (temperature * (1 + delta / 10000)))
    
    elif method == 'fast_drop':
        # More aggressive probability reduction for large deltas
        return math.exp(-(delta ** 1.5) / temperature)
    
    return math.exp(-delta / temperature)  # Default to Boltzmann

def optimized_simulated_annealing(rows, cols, costs, coverage):
    """
    Simulated annealing algorithm for the Set Covering Problem with tuned parameters.
    
    This function uses the optimized parameters found through grid search:
    - max_iterations: 200000
    - initial_temp: 5000
    - no_improve_limit: 15000
    - repair_frequency: 0.2
    - alpha: 130000
    - beta: 25000
    - move_weights: {'flip': 0.3, 'swap': 0.25, 'add': 0.2, 'remove': 0.15, 'cluster': 0.1}
    """
    # Parameter settings from tuning
    max_iterations = 200000
    initial_temp = 5000
    min_temp = 1e-6
    no_improve_limit = 15000
    repair_frequency = 0.2
    alpha = 130000
    beta = 25000
    move_weights = {'flip': 0.3, 'swap': 0.25, 'add': 0.2, 'remove': 0.15, 'cluster': 0.1}
    dynamic_weights = True
    cooling_schedule = 'adaptive'
    acceptance_method = 'boltzmann'
    
    # Start with a greedy initial solution
    current_solution = random_adaptive_solution(rows, cols, costs, coverage)
    
    # Calculate initial cost
    current_total_cost, current_base_cost, uncovered, overcovered = calculate_cost(
        current_solution, costs, coverage, rows, alpha, beta)
    
    # Track best solution
    best_solution = current_solution.copy()
    best_cost = current_total_cost if is_feasible(current_solution, coverage, rows) else float('inf')
    best_base_cost = current_base_cost
    
    # Initialize tracking variables
    temperature = initial_temp
    iteration = 0
    no_improve_counter = 0
    best_feasible_found = False
    local_adjustment_phase = False
    
    # For tracking move success rates
    move_success = {move: 1 for move in move_weights}
    move_attempts = {move: 1 for move in move_weights}
    
    # For adaptive penalty weights
    current_alpha = alpha
    current_beta = beta
    
    while iteration < max_iterations and temperature > min_temp and no_improve_counter < no_improve_limit:
        iteration += 1
        
        # Periodically adjust penalty weights based on solution status
        if iteration % 1000 == 0:
            # If we have many uncovered rows, increase that penalty
            if uncovered > 0:
                current_alpha *= 1.05
            elif uncovered == 0:
                current_alpha = max(alpha, current_alpha * 0.95)  # Gradually decrease but keep minimum
                
            # Similarly for overcovered rows
            if overcovered > 0:
                current_beta *= 1.05
            elif overcovered == 0:
                current_beta = max(beta, current_beta * 0.95)
        
        # Select move strategy based on dynamic weights if enabled
        if dynamic_weights:
            # Calculate success rates for each move
            success_rates = {move: move_success[move] / move_attempts[move] for move in move_weights}
            
            # Normalize to create a probability distribution
            total_rate = sum(success_rates.values())
            if total_rate > 0:
                probs = {move: rate / total_rate for move, rate in success_rates.items()}
                
                # Select move based on calculated probabilities
                r = random.random()
                cumulative = 0
                selected_move = list(probs.keys())[0]  # Default
                for move, prob in probs.items():
                    cumulative += prob
                    if r <= cumulative:
                        selected_move = move
                        break
            else:
                selected_move = random.choice(list(move_weights.keys()))
        else:
            # Use fixed weights
            r = random.random()
            cumulative = 0
            selected_move = list(move_weights.keys())[0]  # Default
            for move, weight in move_weights.items():
                cumulative += weight
                if r <= cumulative:
                    selected_move = move
                    break
        
        # Generate neighbor using selected move strategy
        neighbor = generate_neighbor(current_solution, coverage, rows, selected_move)
        move_attempts[selected_move] += 1
        
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
            temp = adaptive_cooling_schedule(initial_temp, iteration, max_iterations, cooling_schedule)
            prob = acceptance_probability(current_total_cost, new_total_cost, temp, acceptance_method)
            accept = random.random() < prob
        
        if accept:
            # Update current solution
            current_solution = neighbor
            current_total_cost = new_total_cost
            current_base_cost = new_base_cost
            uncovered = new_uncovered
            overcovered = new_overcovered
            
            # Record successful move
            move_success[selected_move] += 1
            
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
        
        # Update temperature using the selected cooling schedule
        temperature = adaptive_cooling_schedule(initial_temp, iteration, max_iterations, cooling_schedule)
        
        # Local adjustment phase - when temperature is very low, focus on minor improvements
        if temperature < initial_temp * 0.01 and not local_adjustment_phase:
            # Switch to more focused local search moves
            move_weights = {'flip': 0.1, 'swap': 0.5, 'add': 0.1, 'remove': 0.2, 'cluster': 0.1}
            local_adjustment_phase = True
        
        # Output progress periodically
        # if iteration % 10000 == 0:
        #     print(f"Iteration {iteration}, Temp: {temperature:.4f}, Current cost: {current_base_cost}, " 
        #           f"Best: {best_cost if best_feasible_found else 'None'}, "
        #           f"Uncovered: {uncovered}, Overcovered: {overcovered}")
    
    # If the best solution found is not feasible, repair it
    if not best_feasible_found or not is_feasible(best_solution, coverage, rows):
        # Try to repair the current solution
        repaired_solution = repair_solution(current_solution, costs, coverage, rows)
        repaired_cost = sum(costs[i] for i in range(len(repaired_solution)) if repaired_solution[i] == 1)
        
        # If repair succeeded and gave a better cost, use it
        if is_feasible(repaired_solution, coverage, rows) and (not best_feasible_found or repaired_cost < best_cost):
            best_solution = repaired_solution
            best_cost = repaired_cost
            best_feasible_found = True
    
    # Return best feasible solution found, or final solution if none was feasible
    if best_feasible_found:
        return best_solution, best_cost
    else:
        # Last attempt to get a feasible solution
        final_solution = repair_solution(current_solution, costs, coverage, rows)
        final_cost = sum(costs[i] for i in range(len(final_solution)) if final_solution[i] == 1)
        return final_solution, final_cost

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

def evaluate_algorithm(file_path, num_trials=30, max_iterations=200000):
    """Evaluates the simulated annealing algorithm across multiple trials."""
    successful_runs = 0
    total_costs = []
    execution_times = []
    
    rows, cols, costs, coverage = parse_problem(file_path)
    
    for trial in range(num_trials):
        print(f"Trial {trial+1}/{num_trials}...")
        start_time = time.time()
        solution, cost = optimized_simulated_annealing(rows, cols, costs, coverage)
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


benchmark_files = ["datasets/sppnw41.txt", "datasets/sppnw42.txt", "datasets/sppnw43.txt"]

# Example usage - run on a single dataset
file_path = 'datasets/sppnw41.txt'
rows, cols, costs, coverage = parse_problem(file_path)
print(f"Running optimized simulated annealing on {file_path}")
best_solution, best_cost = optimized_simulated_annealing(rows, cols, costs, coverage)
print_solution(best_solution, costs, coverage, rows)

# # Run evaluation on all benchmark files with fewer trials for demonstration
# print("\nEvaluating algorithm on all benchmark datasets...")
# benchmark_results = {file: evaluate_algorithm(file) for file in benchmark_files}
    
# # Display results in a table
# df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
# print(df_results)