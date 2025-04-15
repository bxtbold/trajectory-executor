import random
import numpy as np
from trajectory_executor import RobotArmTrajectoryExecutor


current_pos = [0.0, 0.0, 0.0]


def dummy_feedback():
    return [q + random.uniform(-0.01, 0.01) for q in current_pos]


def send_command(q_cmd):
    global current_pos
    current_pos = q_cmd  # setting the command directly for simulation


def monitor(cmd, feedback, t):
    error = np.linalg.norm(np.array(cmd) - np.array(feedback))
    print(f"[{t:.2f}s] Error: {error:.4f}")


executor = RobotArmTrajectoryExecutor(
    update_callback=send_command,
    feedback_callback=dummy_feedback,
    on_feedback=monitor,
)

traj = [(0.0, [0.0, 0.0, 0.0]), (1.5, [0.5, 0.5, 0.5]), (3.0, [1.0, 1.0, 1.0])]
executor.execute(traj)
