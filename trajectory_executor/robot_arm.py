import time
from typing import List, Tuple, Callable, Optional
import numpy as np


class RobotArmTrajectoryExecutor:
    def __init__(
        self,
        trajectory: List[Tuple[float, List[float]]],
        update_callback: Optional[Callable[[List[float]], None]] = None,
        feedback_callback: Optional[Callable[[], List[float]]] = None,
        on_feedback: Optional[Callable[[List[float], List[float], float], None]] = None,
        loop_rate_hz: float = 50.0,
    ):
        self.trajectory = sorted(trajectory, key=lambda x: x[0])
        self.update_callback = update_callback
        self.feedback_callback = feedback_callback
        self.on_feedback = on_feedback
        self.loop_rate = 1.0 / loop_rate_hz

    def _interpolate(self, t: float) -> List[float]:
        for i in range(len(self.trajectory) - 1):
            t0, q0 = self.trajectory[i]
            t1, q1 = self.trajectory[i + 1]
            if t0 <= t <= t1:
                ratio = (t - t0) / (t1 - t0)
                q_interp = np.array(q0) + ratio * (np.array(q1) - np.array(q0))
                return q_interp.tolist()
        return self.trajectory[-1][1]

    def start(self):
        start_time = time.time()
        end_time = self.trajectory[-1][0]
        while True:
            current_time = time.time() - start_time
            if current_time > end_time:
                break

            # Compute command
            joint_cmd = self._interpolate(current_time)

            # Send command
            if self.update_callback:
                self.update_callback(joint_cmd)

            # Read feedback
            joint_feedback = (
                self.feedback_callback() if self.feedback_callback else None
            )

            # Run feedback hook
            if self.on_feedback and joint_feedback is not None:
                self.on_feedback(joint_cmd, joint_feedback, current_time)

            time.sleep(self.loop_rate)

        # Send final command
        if self.update_callback:
            self.update_callback(self.trajectory[-1][1])


if __name__ == "__main__":
    import random

    # Simulated robot feedback: slowly follows the command with noise
    current_pos = [0.0, 0.0, 0.0]

    def fake_feedback():
        return [q + random.uniform(-0.01, 0.01) for q in current_pos]

    def send_command(q_cmd):
        global current_pos
        print(f"Command: {q_cmd}")
        current_pos = q_cmd  # pretend robot follows instantly

    def monitor(cmd, feedback, t):
        error = np.linalg.norm(np.array(cmd) - np.array(feedback))
        print(f"[{t:.2f}s] Error: {error:.4f}")

    traj = [(0.0, [0.0, 0.0, 0.0]), (1.5, [0.5, 0.5, 0.5]), (3.0, [1.0, 1.0, 1.0])]

    executor = RobotArmTrajectoryExecutor(
        trajectory=traj,
        update_callback=send_command,
        feedback_callback=fake_feedback,
        on_feedback=monitor,
    )

    executor.start()
