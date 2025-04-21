import pytest
import numpy as np
import time
import threading
from trajectory_executor import TrajectoryExecutor


# Mock RateLimiter class
class MockRateLimiter:
    def __init__(self, frequency):
        self.frequency = frequency

    def sleep(self):
        # Simulate a short sleep to speed up tests
        time.sleep(0.001 / self.frequency)


# Fixture to create a basic executor
@pytest.fixture
def executor():
    def update_callback(joints: np.ndarray):
        pass  # Default no-op callback

    return TrajectoryExecutor(
        dof=3, update_callback=update_callback, loop_rate_hz=100.0
    )


# Test initialization
def test_init_valid():
    def update_callback(joints: np.ndarray):
        pass

    def feedback_callback() -> np.ndarray:
        return np.array([0.0, 0.0, 0.0])

    def on_feedback(cmd: np.ndarray, fb: np.ndarray, t: float):
        pass

    executor = TrajectoryExecutor(
        dof=2,
        update_callback=update_callback,
        feedback_callback=feedback_callback,
        on_feedback=on_feedback,
        loop_rate_hz=50.0,
    )
    assert executor.dof == 2
    assert executor.has_callbacks["update"] is True
    assert executor.has_callbacks["feedback"] is True
    assert executor.has_callbacks["on_feedback"] is True
    assert executor.loop_rate_hz == 50.0


def test_init_no_callbacks():
    executor = TrajectoryExecutor(
        dof=3, update_callback=lambda x: None, feedback_callback=None, on_feedback=None
    )
    assert executor.has_callbacks["update"] is True
    assert executor.has_callbacks["feedback"] is False
    assert executor.has_callbacks["on_feedback"] is False


# Test interpolation
def test_interpolate_before_trajectory(executor):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    times = np.array([1.0, 2.0])
    result = executor._interpolate(0.0, points, times)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


def test_interpolate_after_trajectory(executor):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    times = np.array([1.0, 2.0])
    result = executor._interpolate(3.0, points, times)
    np.testing.assert_array_equal(result, [4.0, 5.0, 6.0])


def test_interpolate_within_trajectory(executor):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    times = np.array([0.0, 1.0])
    result = executor._interpolate(0.5, points, times)
    np.testing.assert_array_almost_equal(result, [2.5, 3.5, 4.5])


def test_interpolate_at_point(executor):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    times = np.array([0.0, 1.0])
    result = executor._interpolate(0.0, points, times)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])


# Test execute method
def test_execute_empty_trajectory(executor):
    # pass
    points = np.array([], dtype=np.float64).reshape(0, 3)
    times = np.array([], dtype=np.float64)
    executor.loop_rate = MockRateLimiter(100.0)
    executor.execute(points, times)  # Should not raise or call callbacks


def test_execute_single_point():
    commands = []

    def update_callback(joints: np.ndarray):
        commands.append(joints.copy())

    executor = TrajectoryExecutor(
        dof=3, update_callback=update_callback, loop_rate_hz=100.0
    )
    executor.loop_rate = MockRateLimiter(100.0)
    points = np.array([[1.0, 2.0, 3.0]])
    times = np.array([0.0])
    executor.execute(points, times)
    assert len(commands) == 1
    np.testing.assert_array_equal(commands[0], [1.0, 2.0, 3.0])


def test_execute_multiple_points():
    commands = []

    def update_callback(joints: np.ndarray):
        commands.append(joints.copy())

    executor = TrajectoryExecutor(
        dof=3, update_callback=update_callback, loop_rate_hz=100.0
    )
    executor.loop_rate = MockRateLimiter(100.0)
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    times = np.array([0.0, 0.1])
    executor.execute(points, times)
    assert len(commands) >= 2  # At least start and end points
    np.testing.assert_array_almost_equal(commands[0], [1.0, 2.0, 3.0], decimal=3)
    np.testing.assert_array_equal(commands[-1], [4.0, 5.0, 6.0])


def test_execute_unsorted_times():
    commands = []

    def update_callback(joints: np.ndarray):
        commands.append(joints.copy())

    executor = TrajectoryExecutor(
        dof=3, update_callback=update_callback, loop_rate_hz=100.0
    )
    executor.loop_rate = MockRateLimiter(100.0)
    points = np.array([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])  # Unsorted
    times = np.array([0.1, 0.0])
    executor.execute(points, times)
    assert len(commands) >= 2
    np.testing.assert_array_almost_equal(commands[0], [1.0, 2.0, 3.0], decimal=3)
    np.testing.assert_array_equal(commands[-1], [4.0, 5.0, 6.0])


def test_execute_with_feedback():
    commands = []
    feedbacks = []
    times_recorded = []

    def update_callback(joints: np.ndarray):
        commands.append(joints.copy())

    def feedback_callback() -> np.ndarray:
        return np.array([0.1, 0.2, 0.3])

    def on_feedback(cmd: np.ndarray, fb: np.ndarray, t: float):
        feedbacks.append(fb.copy())
        times_recorded.append(t)

    executor = TrajectoryExecutor(
        dof=3,
        update_callback=update_callback,
        feedback_callback=feedback_callback,
        on_feedback=on_feedback,
        loop_rate_hz=100.0,
    )
    executor.loop_rate = MockRateLimiter(100.0)
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    times = np.array([0.0, 0.1])
    executor.execute(points, times)
    assert len(commands) >= 2
    assert len(feedbacks) >= 1
    assert len(times_recorded) >= 1
    np.testing.assert_array_almost_equal(feedbacks[0], [0.1, 0.2, 0.3])
    assert all(0.0 <= t <= 0.1 for t in times_recorded)


def test_execute_thread_safety():
    commands = []
    lock = threading.Lock()

    def update_callback(joints: np.ndarray):
        with lock:
            commands.append(joints.copy())

    executor = TrajectoryExecutor(
        dof=3, update_callback=update_callback, loop_rate_hz=100.0
    )
    executor.loop_rate = MockRateLimiter(100.0)
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    times = np.array([0.0, 0.1])

    # Run execute in a separate thread
    thread = threading.Thread(target=executor.execute, args=(points, times))
    thread.start()
    thread.join()

    assert len(commands) >= 2
    np.testing.assert_array_almost_equal(commands[0], [1.0, 2.0, 3.0], decimal=3)
    np.testing.assert_array_equal(commands[-1], [4.0, 5.0, 6.0])


# Test error handling
def test_execute_mismatched_shapes(executor):
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    times = np.array([0.0])  # Mismatched length
    with pytest.raises(
        IndexError, match="points and times must have the same number of elements"
    ):
        executor.execute(points, times)


def test_execute_invalid_dimensions(executor):
    points = np.array([1.0, 2.0, 3.0])  # 1D instead of 2D
    times = np.array([0.0])
    with pytest.raises(
        IndexError, match="points must be a 2D array and times must be a 1D array"
    ):
        executor.execute(points, times)


def test_execute_wrong_dof(executor):
    points = np.array([[1.0, 2.0]])  # 2 DOF instead of 3
    times = np.array([0.0])
    with pytest.raises(IndexError, match="points must have 3 columns"):
        executor.execute(points, times)


def test_execute_non_numpy_input():
    commands = []

    def update_callback(command: np.ndarray):
        commands.append(command.copy())

    executor = TrajectoryExecutor(
        dof=3, update_callback=update_callback, loop_rate_hz=100.0
    )
    executor.loop_rate = MockRateLimiter(100.0)
    points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]  # List input
    times = [0.0, 0.1]  # List input
    executor.execute(points, times)
    assert len(commands) >= 2
    np.testing.assert_array_almost_equal(commands[0], [1.0, 2.0, 3.0], decimal=3)
    np.testing.assert_array_equal(commands[-1], [4.0, 5.0, 6.0])


# Test RateLimiter integration
def test_execute_rate_limited():
    start_times = []

    def update_callback(joints: np.ndarray):
        start_times.append(time.time())

    executor = TrajectoryExecutor(
        dof=3, update_callback=update_callback, loop_rate_hz=100.0
    )
    executor.loop_rate = MockRateLimiter(100.0)
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    times = np.array([0.0, 0.1])
    executor.execute(points, times)
    assert len(start_times) >= 2
    # Check that updates are roughly 10ms apart (100Hz)
    intervals = np.diff(start_times)
    assert all(0.005 < interval < 0.015 for interval in intervals)  # Allow some jitter
