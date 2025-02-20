import pandas as pd

def get_best_parameters(file_path):
    # Load the parameter tuning results
    df = pd.read_csv(file_path)

    # Sort by the lowest average cost
    best_params = df.sort_values(by="Average Cost").iloc[0]

    # Extract individual parameters
    success_rate = best_params["Success Rate (%)"]
    average_cost = best_params["Average Cost"]
    std_dev = best_params["Standard Deviation"]
    avg_time = best_params["Average Execution Time (s)"]
    params = eval(best_params["Parameters"])
    dataset = best_params["Dataset"]

    # Print the best parameters
    print("========== Best Parameters Found ==========")
    print(f"Success Rate (%): {success_rate}")
    print(f"Average Cost: {average_cost}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Average Execution Time (s): {avg_time}")
    print("Tuned Parameters:")
    print(f"    Initial Temperature: {params[0]}")
    print(f"    Cooling Rate: {params[1]}")
    print(f"    Max Iterations: {params[2]}")
    print(f"    Min Temperature: {params[3]}")
    print(f"    No Improvement Limit: {params[4]}")
    print(f"    Penalty Factor: {params[5]}")
    print(f" Dataset Used: {dataset}")
    print("===========================================")

    return best_params

# Example usage
if __name__ == "__main__":
    file_path = "parameter_tuning_results_all.csv"  # Change if needed
    best_params = get_best_parameters(file_path)
