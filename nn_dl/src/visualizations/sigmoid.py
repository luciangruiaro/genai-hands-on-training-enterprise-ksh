import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Generate x values
x = np.linspace(-10, 10, 400)

# Compute sigmoid values for each x
y = sigmoid(x)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="Sigmoid Function")
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True)
plt.legend()
plt.show()
