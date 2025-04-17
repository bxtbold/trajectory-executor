import time
import numpy as np
from typing import List, Tuple, Callable, Optional
from loop_rate_limiters import RateLimiter
import threading


class RobotArmTrajectoryExecutor:
    """Executes a trajectory for a robot arm with rate-limited updates and thread-safe callbacks.

    This class manages the execution of a time-based joint trajectory for a robot arm. It interpolates
    joint positions between trajectory points, invokes user-provided callbacks for updates and feedback,
    and ensures thread-safe operations using a lock. The execution is rate-limited to a specified frequency.

    Attributes:
        update_callback (Callable[[List[float]], None]): Callback to send joint commands to the robot.
        feedback_callback (Optional[Callable[[], List[float]]]): Callback to retrieve joint feedback.
        on_feedback (Optional[Callable[[List[float], List[float], float], None]]): Callback to process
            command, feedback, and time.
        loop_rate (RateLimiter): Rate limiter to control update frequency.
        trajectory (List): Current trajectory being executed (list of time and joint positions).
        has_callbacks (dict): Tracks availability of callbacks.
        _lock (threading.Lock): Lock for thread-safe callback execution.

    Args:
        update_callback (Callable[[List[float]], None]): Function to send joint commands to the robot.
        feedback_callback (Optional[Callable[[], List[float]]], optional): Function to get joint feedback.
            Defaults to None.
        on_feedback (Optional[Callable[[List[float], List[float], float], None]], optional): Function to
            handle command, feedback, and time. Defaults to None.
        loop_rate_hz (float, optional): Frequency of updates in Hertz. Defaults to 50.0.
    """

    def __init__(
        self,
        update_callback: Callable[[List[float]], None],
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

    def _interpolate(self, t: float, traj: np.ndarray, times: np.ndarray) -> List[float]:
        """Interpolates joint positions at a given time based on the trajectory.

        Args:
            t (float): Current time for interpolation.
            traj (np.ndarray): Array of joint positions in the trajectory.
            times (np.ndarray): Array of time points corresponding to the trajectory.

        Returns:
            List[float]: Interpolated joint positions at time `t`.
        """
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
        """Executes the provided trajectory by interpolating joint positions and invoking callbacks.

        The trajectory is a list of tuples, each containing a time and a list of joint positions. The method
        interpolates joint positions at the current time, sends commands via the update callback, and handles
        feedback if provided. Execution is rate-limited and thread-safe.

        Args:
            trajectory (List[Tuple[float, List[float]]]): List of (time, joint_positions) tuples defining
                the trajectory.

        Returns:
            None
        """
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
