import pytest
import numpy as np
from trajectory_executor import RobotArmTrajectoryExecutor


# Mock callback for testing
def mock_update_callback(joint_cmd):
    pass


@pytest.mark.skipif(not pytest.importorskip("mujoco"), reason="mujoco not installed")
def test_interpolate_with_mujoco():
    executor = RobotArmTrajectoryExecutor(update_callback=mock_update_callback)
    traj = np.array([[1.0, 2.0], [3.0, 4.0]])
    times = np.array([0.0, 1.0])
    result = executor._interpolate(0.5, traj, times)
    assert result == [2.0, 3.0]  # Linear interpolation at t=0.5


def test_execute_empty_trajectory():
    executor = RobotArmTrajectoryExecutor(update_callback=mock_update_callback)
    executor.execute([])  # Should handle empty trajectory gracefully
    assert True  # No exceptions means success
