import threading
import time
import numpy as np
from typing import Callable, Optional
from loop_rate_limiters import RateLimiter


class TrajectoryExecutor:
    """Executes a trajectory with rate-limited updates and thread-safe callbacks.

    This class manages the execution of a time-based trajectory. It interpolates
    positions between trajectory points, invokes user-provided callbacks for updates and feedback,
    and ensures thread-safe operations using a lock. The execution is rate-limited to a specified frequency.

    Args:
        dof (int): Number of degrees of freedom.
        update_callback (Callable[[np.ndarray], None]): Function to send commands to the robot.
        feedback_callback (Optional[Callable[[], np.ndarray]], optional): Function to get feedback.
            Defaults to None.
        on_feedback (Optional[Callable[[np.ndarray, np.ndarray, float], None]], optional): Function to
            handle command, feedback, and time. Defaults to None.
        loop_rate_hz (float, optional): Frequency of updates in Hertz. Defaults to 50.0.
    """

    def __init__(
        self,
        dof: int,
        update_callback: Callable[[np.ndarray], None],
        feedback_callback: Optional[Callable[[], np.ndarray]] = None,
        on_feedback: Optional[Callable[[np.ndarray, np.ndarray, float], None]] = None,
        loop_rate_hz: float = 50.0,
    ):
        self.dof = dof
        self.update_callback = update_callback
        self.feedback_callback = feedback_callback
        self.on_feedback = on_feedback
        self.loop_rate_hz = loop_rate_hz
        self.has_callbacks = {
            "update": update_callback is not None,
            "feedback": feedback_callback is not None,
            "on_feedback": on_feedback is not None,
        }
        self._lock = threading.Lock()

    def _interpolate(
        self, t: float, points: np.ndarray, times: np.ndarray
    ) -> np.ndarray:
        """Interpolates points at a given time based on the trajectory.

        Args:
            t (float): Current time for interpolation.
            points (np.ndarray): Array of points in the trajectory.
            times (np.ndarray): Array of time points corresponding to the trajectory.

        Returns:
            np.ndarray: Interpolated points at time `t`.
        """
        if t >= times[-1]:
            return points[-1].copy()
        if t <= times[0]:
            return points[0].copy()
        idx = np.searchsorted(times, t, side="right") - 1
        t0, t1 = times[idx], times[idx + 1]
        q0, q1 = points[idx], points[idx + 1]
        ratio = (t - t0) / (t1 - t0)
        return q0 + ratio * (q1 - q0)

    def execute(
        self, points: np.ndarray, times: np.ndarray, wait_until_done: bool = True
    ):
        """Executes the provided trajectory by interpolating points and invoking callbacks.

        The trajectory is array of points and correspoding time to reach to the point. The method
        interpolates positions at the current time, sends commands via the update callback, and handles
        feedback if provided. Execution is rate-limited and thread-safe.

        Args:
            points (np.ndarray): Trajectory points to be executed.
            times (np.ndarray): Time each point to be executed.
            wait_until_done (bool, optional): If True, waits until the trajectory is fully executed.

        Returns:
            None
        """
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)
        if not isinstance(times, np.ndarray):
            times = np.array(times, dtype=np.float64)

        if points.size == 0 or times.size == 0:
            return
        if points.ndim != 2 or times.ndim != 1:
            raise IndexError("points must be a 2D array and times must be a 1D array")
        if points.shape[0] != times.shape[0]:
            raise IndexError("points and times must have the same number of elements")
        if points.shape[1] != self.dof:
            raise IndexError(
                f"points must have {self.dof} columns, but got {points.shape[1]}"
            )

        # Verify trajectory is sorted
        if not np.all(times[:-1] <= times[1:]):
            sorted_indices = np.argsort(times)
            times = times[sorted_indices]
            points = points[sorted_indices]

        start_time = time.time()
        end_time = times[-1]

        loop_rate = RateLimiter(self.loop_rate_hz)
        while True:
            current_time = time.time() - start_time
            if current_time > end_time:
                break

            # Interpolate target state
            target_state = self._interpolate(current_time, points, times)

            # Thread-safe callback execution
            with self._lock:
                if self.has_callbacks["update"]:
                    self.update_callback(target_state)

                # Handle feedback
                if self.has_callbacks["feedback"]:
                    current_state = self.feedback_callback()
                if self.has_callbacks["on_feedback"]:
                    self.on_feedback(target_state, current_state, current_time)

            loop_rate.sleep()

        # Send final command thread-safely
        with self._lock:
            if self.has_callbacks["update"]:
                self.update_callback(points[-1])
