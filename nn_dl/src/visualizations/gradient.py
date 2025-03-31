import numpy as np
import matplotlib.pyplot as plt

# Function and its derivative
def f(x):
    return x**2

def df(x):
    return 2*x

# Gradient Descent
def gradient_descent(learning_rate, epochs, start_x):
    x = start_x
    history = [x]

    for _ in range(epochs):
        gradient = df(x)
        x -= learning_rate * gradient
        history.append(x)

    return history

# Visualization
x = np.linspace(-10, 10, 400)
y = f(x)

start_x = 9  # Starting point
learning_rate = 0.1
epochs = 20

history = gradient_descent(learning_rate, epochs, start_x)
history_y = [f(point) for point in history]

plt.figure(figsize=(10,6))
plt.plot(x, y, label='f(x) = x^2')
plt.scatter(history, history_y, color='red', marker='o')
plt.title('Gradient Descent Visualization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
