import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from typing import List

from trajectory_executor import TrajectoryExecutor


# === 1. Initialize Mujoco ===
model = mujoco.MjModel.from_xml_path("models/universal_robots_ur5e/ur5e.xml")
data = mujoco.MjData(model)


# === 2. Define callbacks and executor ===
def send_joint_command(cmd: List[float]):
    data.ctrl[: len(cmd)] = cmd


def get_joint_feedback() -> List[float]:
    return data.qpos[: model.nu].copy().tolist()


def monitor(cmd, feedback, t):
    error = np.linalg.norm(np.array(cmd) - np.array(feedback))
    print(f"[{t:.2f}s] Tracking error: {error:.4f}")


executor = TrajectoryExecutor(
    dof=6,
    update_callback=send_joint_command,
    feedback_callback=get_joint_feedback,
    on_feedback=monitor,
    loop_rate_hz=100,
)


# === 3. Run viewer in a separate thread ===
def viewer_thread():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
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

# Keep the main thread alive until the viewer closes
viewer_t.join()
