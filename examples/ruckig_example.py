import numpy as np
from ruckig import InputParameter, Ruckig, Trajectory, Result
from trajectory_executor import TrajectoryExecutor


# === 1. Initialize Ruckig ===
inp = InputParameter(6)
inp.current_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
inp.target_position = [0.0, -1.01, -1.7, -1.82, 1.45, 0.0]
inp.max_velocity = [3.14, 3.14, 3.14, 3.14, 3.14, 3.14]
inp.max_acceleration = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
inp.max_jerk = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
inp.minimum_duration = 5.0

otg = Ruckig(6)
trajectory = Trajectory(6)

# === 2. Calculate trajectory ===
result = otg.calculate(inp, trajectory)
if result == Result.ErrorInvalidInput:
    raise Exception("Invalid input!")

times = []
positions = []

for new_time in np.linspace(0, trajectory.duration, 10):
    # Then, we can calculate the kinematic state at a given time
    pos, _, _ = trajectory.at_time(new_time)
    times.append(new_time)
    positions.append(pos)

# === 3. Define callback and trajectory executor ===
def update_callback(target_state):
    print(f"Target position: {target_state}")

executor = TrajectoryExecutor(
    dof=trajectory.degrees_of_freedom,
    update_callback=update_callback,
    loop_rate_hz=50.0,
)

# === 4. Execute the trajectory ===
executor.execute(positions, times)
