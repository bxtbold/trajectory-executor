# trajectory-executor

<!-- [![Build](https://img.shields.io/github/actions/workflow/status/bxtbold/trajectory-executor/ci.yml?branch=main)](https://github.com/bxtbold/trajectory-executor/actions) -->
<!-- [![PyPI version](https://img.shields.io/pypi/v/trajectory-executor)](https://pypi.org/project/trajectory-executor/) -->
<!-- [![Documentation](https://img.shields.io/github/actions/workflow/status/bxtbold/trajectory-executor/docs.yml?branch=main&label=docs)](https://bxtbold.github.io/trajectory-executor/) -->

Execute time-based joint trajectories for robot arms with rate-limited, thread-safe updates.

## Installation

```console
pip install trajectory-executor
```

or

```console
git clone https://github.com/bxtbold/trajectory-executor.git
cd trajectory-executor
pip install -e .
```

### Dependencies

- Python 3.10+
- `numpy` (>=1.20.0)
- `loop-rate-limiters` (>=0.1.0)

## Usage

The `RobotArmTrajectoryExecutor` class executes joint trajectories for a robot arm with specified degrees of freedom (DOF). It interpolates positions, sends commands via a callback, and supports optional feedback processing.

```python
import numpy as np
from trajectory-executor import RobotArmTrajectoryExecutor

def update_callback(joints: np.ndarray):
    print(f"Joint command: {joints}")

executor = RobotArmTrajectoryExecutor(dof=3, update_callback=update_callback)
points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
times = np.array([0.0, 1.0])
executor.execute(points, times)
```

## Testing

Run the test suite with `pytest`:

```console
pip install pytest
pytest tests/test_robot_arm_trajectory-executor.py -v
```

## Contributing

1. Fork and clone: `git clone https://github.com/bxtbold/trajectory-executor.git`
2. Create a branch: `git checkout -b feature/your-feature`
3. Add changes and tests in `tests/`
4. Run tests: `pytest`
5. Submit a pull request

## License

MIT License. See the [LICENSE](LICENSE) file.
