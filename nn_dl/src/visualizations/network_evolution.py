import json
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":
    json_file_path = sys.argv[1]
    target_number = int(sys.argv[2])

    with open(json_file_path, 'r') as file:
        predicted_outputs = json.load(file)

    x_vals = range(1, len(predicted_outputs) + 1)

    # If target_number is -1, plot all numbers, otherwise only plot the target number
    if target_number != -1:
        y_vals = [data[target_number] for data in predicted_outputs]
        plt.plot(x_vals, y_vals, '-o', label=f"Number {target_number}")
    else:
        for j in range(10):  # Assuming we have 10 possible numbers (0-9)
            y_vals = [data[j] for data in predicted_outputs]
            plt.plot(x_vals, y_vals, '-o', label=f"Number {j}")

    # Setting graph attributes
    plt.ylim(0, 1)
    plt.xlabel('Training step')
    plt.ylabel('Probability')
    plt.title('Predicted Outputs per Training step')
    plt.legend(loc="best")

    plt.grid(True, which='both', linestyle='--', linewidth=0.1)
    plt.tight_layout()
    plt.show()
