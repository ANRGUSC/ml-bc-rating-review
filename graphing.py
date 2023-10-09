import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Constants
NUM_USER_POINTS = 10

# Initialization
def initialize_points():
    model_point = (random.uniform(0, 10), random.uniform(0, 10), 0)
    expert_point = (random.uniform(0, 10), random.uniform(0, 10))
    user_points = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(NUM_USER_POINTS)]
    
    return model_point, expert_point, user_points

model_point, expert_point, user_points = initialize_points()


def update_model_point(model_point):
    """Update the position of the model_point and the round number."""
    # Randomly move the point within a range of [-0.5, 0.5] for both x and y coordinates
    new_x = model_point[0] + random.uniform(-0.5, 0.5)
    new_y = model_point[1] + random.uniform(-0.5, 0.5)
    # Increase the round number by 1
    new_round = model_point[2] + 1
    return [new_x, new_y, new_round]

# Visualization
def plot_all_points_movement(model_point_history, user_points, expert_point):
    """Plot the movement of the model point, user points, and expert point over rounds."""
    plt.figure(figsize=(10, 7))
    
    # Plot movement of model point
    xs = [point[0] for point in model_point_history]
    ys = [point[1] for point in model_point_history]
    rounds = [point[2] for point in model_point_history]
    plt.scatter(xs, ys, c=rounds, cmap='viridis', s=100, alpha=0.8, label="Model Point")
    plt.plot(xs, ys, '-o', alpha=0.6)
    
    # Plot user points
    user_xs = [point[0] for point in user_points]
    user_ys = [point[1] for point in user_points]
    plt.scatter(user_xs, user_ys, c='blue', s=100, alpha=0.6, label='User Points')
    
    # Plot expert point
    plt.scatter(*expert_point, c='green', s=150, alpha=0.8, marker='*', label='Expert Point')
    
    plt.colorbar().set_label('Round Number for Model Point')
    plt.title("Movement of Model Point and Position of User and Expert Points")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.show()

def animate_all_points_movement(model_point_history, user_points, expert_point):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title("Movement of Model Point and Position of User and Expert Points")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)

    # Plot user points and expert point (they remain static)
    user_xs = [point[0] for point in user_points]
    user_ys = [point[1] for point in user_points]
    ax.scatter(user_xs, user_ys, c='blue', s=100, alpha=0.6, label='User Points')
    ax.scatter(*expert_point, c='green', s=150, alpha=0.8, marker='*', label='Expert Point')

    # Model point's trace and current position
    xs = [point[0] for point in model_point_history]
    ys = [point[1] for point in model_point_history]
    line, = ax.plot([], [], '-o', color='purple', alpha=0.6)
    point, = ax.plot([], [], 'o', color='red', alpha=0.8, markersize=10, label='Model Point')

    # Initialization function
    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    # Animation update function with modification to address the warning
    def update(frame):
        line.set_data(xs[:frame+1], ys[:frame+1])
        point.set_data([xs[frame]], [ys[frame]])  # Wrap values in a list
        return line, point

    ani = FuncAnimation(fig, update, frames=len(xs), init_func=init, blit=True, repeat=False)
    plt.show()



def main():
    model_point, expert_point, user_points = initialize_points()
    model_point_history = [model_point]
    for _ in range(10):
        model_point = update_model_point(model_point)
        model_point_history.append(model_point)

    animate_all_points_movement(model_point_history, user_points, expert_point)

if __name__ == "__main__":
    main()
    

    
