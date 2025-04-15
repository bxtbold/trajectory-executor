import time
import numpy as np
from typing import List, Tuple, Callable, Optional
from loop_rate_limiters import RateLimiter
import threading


class RobotArmTrajectoryExecutor:
    def __init__(
        self,
        update_callback: Optional[Callable[[List[float]], None]] = None,
        feedback_callback: Optional[Callable[[], List[float]]] = None,
        on_feedback: Optional[Callable[[List[float], List[float], float], None]] = None,
        loop_rate_hz: float = 50.0,
    ):
        self.update_callback = update_callback
        self.feedback_callback = feedback_callback
        self.on_feedback = on_feedback
        self.loop_rate = RateLimiter(loop_rate_hz)
        self.trajectory = []
        self.has_callbacks = {
            "update": update_callback is not None,
            "feedback": feedback_callback is not None,
            "on_feedback": on_feedback is not None,
        }
        self._lock = threading.Lock()  # Lock for thread-safe access to shared resources

    def _interpolate(
        self, t: float, traj: np.ndarray, times: np.ndarray
    ) -> List[float]:
        if t >= times[-1]:
            return traj[-1].tolist()
        idx = np.searchsorted(times, t, side="right") - 1
        if idx < 0:
            return traj[0].tolist()
        t0, t1 = times[idx], times[idx + 1]
        q0, q1 = traj[idx], traj[idx + 1]
        ratio = (t - t0) / (t1 - t0)
        return (q0 + ratio * (q1 - q0)).tolist()

    def execute(
        self,
        trajectory: List[Tuple[float, List[float]]],
    ):
        if not trajectory:
            return

        # Copy trajectory to prevent external modifications
        trajectory = [(t, q.copy()) for t, q in trajectory]

        # Convert trajectory to NumPy arrays for efficiency
        times = np.array([t for t, _ in trajectory])
        traj = np.array([q for _, q in trajectory])

        # Verify trajectory is sorted
        if not np.all(times[:-1] <= times[1:]):
            sorted_indices = np.argsort(times)
            times = times[sorted_indices]
            traj = traj[sorted_indices]

        start_time = time.time()
        end_time = times[-1]

        while True:
            current_time = time.time() - start_time
            if current_time > end_time:
                break

            # Compute command
            joint_cmd = self._interpolate(current_time, traj, times)

            # Thread-safe callback execution
            with self._lock:
                if self.has_callbacks["update"]:
                    self.update_callback(joint_cmd)

                # Handle feedback
                if self.has_callbacks["feedback"] and self.has_callbacks["on_feedback"]:
                    joint_feedback = self.feedback_callback()
                    self.on_feedback(joint_cmd, joint_feedback, current_time)

            self.loop_rate.sleep()

        # Send final command thread-safely
        with self._lock:
            if self.has_callbacks["update"]:
                self.update_callback(traj[-1].tolist())


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

    traj_executor = RobotArmTrajectoryExecutor(
        update_callback=send_command,
        feedback_callback=fake_feedback,
        on_feedback=monitor,
    )

    traj = [(0.0, [0.0, 0.0, 0.0]), (1.5, [0.5, 0.5, 0.5]), (3.0, [1.0, 1.0, 1.0])]
    traj_executor.execute(traj)
