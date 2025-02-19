import random
import math
import time
import numpy as np
import pandas as pd

def parse_problem(file_path):
    with open(file_path, 'r') as f:
        rows, cols = map(int, f.readline().split())
        # print(rows, cols)
    
        costs, coverage = [], []

        for _ in range(cols):
            parts = list(map(int, f.readline().split()))
            
            cost = parts[0]
            costs.append(cost)

            num_flights = parts[1]
            flights = set(parts[2 : 2 + num_flights])
            coverage.append(flights)
        
        # print(costs)
        # print('\n')
        # print(coverage)
        return rows, cols, costs, coverage

# parse_problem('datasets/sppnw41.txt')

def calculate_cost(solution, costs,  coverage, rows, penality_factor=10000):
    total_cost = sum(costs[i] for i in range(len(solution)) if solution[i] == 1)

    flight_coverage = [0] * rows

    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight - 1] += 1
    
    penality = sum(abs(count -1) for count in flight_coverage) * penality_factor

    return total_cost + penality

def simulated_annealing(rows, cols, costs, coverage, max_iter=1000,initial_temp=1000, cooling_rate=0.99, min_temp=1e-3, no_improve_limit=1000):
    #initialize a random feadible solution
    current_solution = [random.randint(0,1) for _ in range(cols)]

    best_solution = current_solution[:]
    best_cost = calculate_cost(best_solution, costs, coverage, rows)

    temperature = initial_temp
    no_improvement_count = 0

    for iteration in range(max_iter):
        if temperature < min_temp or no_improvement_count > no_improve_limit:
            break
        
        # Generate a neighboring solution by flipping a random bit
        neighbour = current_solution[:]
        flip_index = random.randint(0, cols -1)

        neighbour[flip_index] = 1 - neighbour[flip_index]

        # Calculate the cost of the neighbor
        if random.random() < 1 and not is_feasible(neighbour, coverage, rows):
            neighbour = repair_solution(neighbour, costs, coverage, rows)
        neighbour_cost = calculate_cost(neighbour, costs, coverage, rows)

        #acceptance probability
        delta_cost = neighbour_cost - best_cost
        if delta_cost < 0 or random.random() < math.exp(-delta_cost/temperature):
            current_solution = neighbour
            if neighbour_cost < best_cost:
                best_solution = neighbour
                best_cost = neighbour_cost
                no_improvement_count = 0
            else: no_improvement_count +=1
        
        temperature *= cooling_rate

        if no_improvement_count > no_improve_limit:
            break

    return best_solution, best_cost

def is_feasible(solution, coverage, rows):
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight -1] +=1
    
    # print("Flight coverage: ", flight_coverage)

    return all(count >= 1 for count in flight_coverage)


def repair_solution(solution, costs, coverage, rows):
    #identify uncovered flights
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight -1] +=1

    #add missing rotations to cover uncovered flights
    for flight in range(rows):
        if flight_coverage[flight] == 0:
            #find the cheapest rotation that covers this flight
            best_rotation = -1
            best_cost = float('inf')
            for i in range(len(solution)):
                if (flight + 1) in coverage[i] and costs[i] < best_cost:
                    best_rotation = i
                    best_cost = costs[i]
            #add the best rotation to the solution
            if best_rotation != -1:
                solution[best_rotation] = 1
                #update flight coverage
                for f in coverage[best_rotation]:
                    flight_coverage[f -1] +=1
    
    #remove redumdant rotations that over-cover flights
    for i in range(len(solution)):
        if solution[i] == 1:
            #check if removing this rotation still leaves all flights covered
            can_remove = all(flight_coverage[flight -1] > 1 for flight in coverage[i])
            if can_remove:
                solution[i] = 0
                for flight in coverage[i]:
                    flight_coverage[flight - 1] -= 1

    return solution


def print_solution(solution, costs, coverage):
    selected_schedules = [(i, costs[i], coverage[i]) for i in range(len(solution)) if solution[i] == 1]

    print("\n==== Simulated Annealing Solution ====")
    print(f"Total Selected Crew Schedules: {len(selected_schedules)}")

    if not selected_schedules:
        print("No schedules selected")
        return
    
    total_cost = sum(cost for _,cost, _ in selected_schedules)
    print(f"Total Cost: {total_cost}")
    print("\nSelected Crew Schedules:")

    for idx, cost, flights in selected_schedules:
        print(f"- Schedule {idx}: Cost = {cost}, Covers Flights: {sorted(flights)}")
    
    flight_coverage = [0] * rows
    for i in range(len(solution)):
        if solution[i] == 1:
            for flight in coverage[i]:
                flight_coverage[flight -1] +=1
    
    print("====================================\n")
    print("Flight coverage: ", flight_coverage)




rows, cols, costs, coverage = parse_problem("datasets/sppnw42.txt")

best_sa_solution, best_sa_cost = simulated_annealing(rows, cols, costs, coverage)

print_solution(best_sa_solution, costs, coverage)
feasible = is_feasible(best_sa_solution, coverage, rows)
print(feasible)
print("\n")

def evaluate_algorithm(rows, cols, costs, coverage, num_trials=30):

    successful_runs = 0
    total_cost = []
    total_times = []

    for _ in range(num_trials):
        start_time = time.time()

        solution, cost = simulated_annealing(rows, cols, costs, coverage)
        feasible = is_feasible(solution, coverage, rows)
        duration = time.time() - start_time

        if feasible:
            successful_runs +=1
            total_cost.append(cost)

        total_times.append(duration)

    success_rate = (successful_runs / num_trials) * 100
    average_cost = np.mean(total_cost) if total_cost else float('inf')
    std_dev_cost = np.std(total_cost) if total_cost else float('inf')
    average_time = np.mean(total_times)

    return {
        "Success Rate (%)": success_rate,
        "Average Cost": average_cost,
        "Standard Deviation": std_dev_cost,
        "Average Execution Time (s)": average_time
    }

# Load and evaluate each benchmark problem
benchmark_files = ["datasets/sppnw41.txt","datasets/sppnw42.txt","datasets/sppnw43.txt"]
benchmark_results = {}

for file_path in benchmark_files:
    rows, cols, costs, coverage = parse_problem(file_path)
    benchmark_results[file_path] = evaluate_algorithm(rows, cols, costs, coverage)

# Display results for each benchmark problem
df_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
print(df_results)