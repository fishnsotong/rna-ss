import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

sns.set_style("ticks")

# Generate some synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias column to X (for the intercept term)
X_b = np.c_[np.ones((100, 1)), X]  # X_b is now (100, 2)

# Initialize parameters for gradient descent
theta = np.random.randn(2, 1)  # Start with random values for theta
learning_rate = 0.1
n_iterations = 50
m = len(X_b)
cost_history = []

# Function to compute cost (Mean Squared Error)
def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    return (1 / (2 * m)) * np.sum(np.square(predictions - y))

# Perform gradient descent and store history
theta_history = []
cost_history = []

for iteration in range(n_iterations):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    cost = compute_cost(X_b, y, theta)
    theta_history.append(theta.copy())
    cost_history.append(cost)

# Setup figure and subplots for animation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Scatter plot for data points
ax1.scatter(X, y, color='blue')
line, = ax1.plot([], [], color='red')

# Cost plot
ax2.set_xlim(0, n_iterations)
ax2.set_ylim(0, max(cost_history))
cost_line, = ax2.plot([], [], color='green')

# Function to initialize the animation
def init():
    line.set_data([], [])
    cost_line.set_data([], [])
    return line, cost_line

# Function to update the animation
def update(frame):
    # Update regression line using theta from the current frame
    theta = theta_history[frame]
    
    # Use full X_b for prediction to include bias term
    line_x = np.linspace(0, 2, 100)
    line_x_b = np.c_[np.ones((100, 1)), line_x]  # Add bias term to line_x
    line_y = line_x_b.dot(theta)  # Predict using current theta
    
    line.set_data(line_x, line_y)
    
    # Update cost plot
    cost_line.set_data(range(frame + 1), cost_history[:frame + 1])
    
    # Titles and labels
    ax1.set_title(f'Iteration {frame}: Linear Regression Line')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax2.set_title('Gradient Descent Progress (Cost vs Iterations)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Cost (MSE)')
    
    return line, cost_line

# Create animation
ani = FuncAnimation(fig, update, frames=n_iterations, init_func=init, blit=False, interval=200)

# Show animation
plt.tight_layout()
plt.show()