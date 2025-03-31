import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2

def df(x):
    return 2*x

def gradient_descent(learning_rate, epochs, start_x):
    x = start_x
    history = [x]

    for _ in range(epochs):
        gradient = df(x)
        x -= learning_rate * gradient
        history.append(x)

    return history

x = np.linspace(-10, 10, 400)
y = f(x)

start_x = 9
epochs = 20
learning_rates = [0.01, 0.1, 0.9]

plt.figure(figsize=(12,8))

for lr in learning_rates:
    history = gradient_descent(lr, epochs, start_x)
    history_y = [f(point) for point in history]
    plt.plot(history, history_y, label=f'Learning Rate: {lr}', marker='o')

plt.plot(x, y, label='f(x) = x^2', color='black', alpha=0.6)
plt.title('Impact of Learning Rate on Gradient Descent')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()
