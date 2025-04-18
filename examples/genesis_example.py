import genesis as gs
import numpy as np
import threading
import time
import torch
from typing import List

from trajectory_executor import RobotArmTrajectoryExecutor


# === 1. Initialize Genesis ===
device = gs.gpu if torch.cuda.is_available() else gs.cpu
gs.init(backend=device, logging_level="error")

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(max_FPS=60),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)
ur5e = scene.add_entity(
    gs.morphs.MJCF(file="models/universal_robots_ur5e/ur5e.xml"),
)
scene.build()

# set ur5e gains
ur5e.set_dofs_kp(np.array([2500, 2500, 2500, 1250, 1250, 1250]))
ur5e.set_dofs_kv(np.array([500, 500, 500, 250, 250, 250]))
ur5e.set_dofs_force_range(
    np.array([-300, -300, -300, -100, -100, -100]),
    np.array([300, 300, 300, 100, 100, 100]),
)


# === 2. Define callbacks and executor ===
def send_joint_command(cmd: List[float]):
    ur5e.control_dofs_position(cmd)


def get_joint_feedback() -> List[float]:
    return ur5e.get_qpos().cpu()


def monitor(cmd, feedback, t):
    error = np.linalg.norm(np.array(cmd) - np.array(feedback))
    print(f"[{t:.2f}s] Tracking error: {error:.4f}")


executor = RobotArmTrajectoryExecutor(
    dof=6,
    update_callback=send_joint_command,
    feedback_callback=get_joint_feedback,
    on_feedback=monitor,
    loop_rate_hz=100,
)


# === 3. Run viewer in a separate thread ===
def viewer_thread():
    while scene.viewer.is_alive():
        scene.step()
        time.sleep(0.01)


# Start the viewer thread
viewer_t = threading.Thread(target=viewer_thread, daemon=True)
viewer_t.start()

# === 4. Define and execute a simple trajectory ===
times = [0.0, 2.5, 5.0]
points = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -0.6, -0.5, -1.0, 0.8, 0.0],
    [0.0, -1.01, -1.7, -1.82, 1.45, 0.0],
]
executor.execute(times=times, points=points)

viewer_t.join()
