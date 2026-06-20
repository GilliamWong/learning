# Gymnasium policy scaffolds

38 environments scaffolded across 4 families.
Each file is a runnable random-policy template — implement `Policy.act()`.

### box2d/
- `box2d/BipedalWalker_v3.py` — `BipedalWalker-v3`
- `box2d/BipedalWalkerHardcore_v3.py` — `BipedalWalkerHardcore-v3`
- `box2d/CarRacing_v3.py` — `CarRacing-v3`
- `box2d/LunarLander_v3.py` — `LunarLander-v3`
- `box2d/LunarLanderContinuous_v3.py` — `LunarLanderContinuous-v3`

### classic_control/
- `classic_control/Acrobot_v1.py` — `Acrobot-v1`
- `classic_control/CartPole_v0.py` — `CartPole-v0`
- `classic_control/CartPole_v1.py` — `CartPole-v1`
- `classic_control/MountainCar_v0.py` — `MountainCar-v0`
- `classic_control/MountainCarContinuous_v0.py` — `MountainCarContinuous-v0`
- `classic_control/Pendulum_v1.py` — `Pendulum-v1`

### mujoco/
- `mujoco/Ant_v4.py` — `Ant-v4`
- `mujoco/Ant_v5.py` — `Ant-v5`
- `mujoco/HalfCheetah_v4.py` — `HalfCheetah-v4`
- `mujoco/HalfCheetah_v5.py` — `HalfCheetah-v5`
- `mujoco/Hopper_v4.py` — `Hopper-v4`
- `mujoco/Hopper_v5.py` — `Hopper-v5`
- `mujoco/Humanoid_v4.py` — `Humanoid-v4`
- `mujoco/Humanoid_v5.py` — `Humanoid-v5`
- `mujoco/HumanoidStandup_v4.py` — `HumanoidStandup-v4`
- `mujoco/HumanoidStandup_v5.py` — `HumanoidStandup-v5`
- `mujoco/InvertedDoublePendulum_v4.py` — `InvertedDoublePendulum-v4`
- `mujoco/InvertedDoublePendulum_v5.py` — `InvertedDoublePendulum-v5`
- `mujoco/InvertedPendulum_v4.py` — `InvertedPendulum-v4`
- `mujoco/InvertedPendulum_v5.py` — `InvertedPendulum-v5`
- `mujoco/Pusher_v5.py` — `Pusher-v5`
- `mujoco/Reacher_v4.py` — `Reacher-v4`
- `mujoco/Reacher_v5.py` — `Reacher-v5`
- `mujoco/Swimmer_v4.py` — `Swimmer-v4`
- `mujoco/Swimmer_v5.py` — `Swimmer-v5`
- `mujoco/Walker2d_v4.py` — `Walker2d-v4`
- `mujoco/Walker2d_v5.py` — `Walker2d-v5`

### toy_text/
- `toy_text/Blackjack_v1.py` — `Blackjack-v1`
- `toy_text/CliffWalking_v1.py` — `CliffWalking-v1`
- `toy_text/CliffWalkingSlippery_v1.py` — `CliffWalkingSlippery-v1`
- `toy_text/FrozenLake_v1.py` — `FrozenLake-v1`
- `toy_text/FrozenLake8x8_v1.py` — `FrozenLake8x8-v1`
- `toy_text/Taxi_v4.py` — `Taxi-v4`

### Not generated (need extra installs)
These are registered but couldn't be created in this environment (e.g. MuJoCo needs `pip install gymnasium[mujoco]`, Atari needs `pip install gymnasium[atari] ale-py`). Re-run `python _tools/gen_gym_scaffolds.py` after installing to add them:

- `Ant-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Ant-v3` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `HalfCheetah-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `HalfCheetah-v3` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Hopper-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Hopper-v3` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Humanoid-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Humanoid-v3` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `HumanoidStandup-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `InvertedDoublePendulum-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `InvertedPendulum-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Pusher-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Pusher-v4` — ImportError: `Pusher-v4` is only supported on `mujoco<3`, for more information https://github.com/Farama-Foundation/Gymnasium/issues/950
- `Reacher-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Swimmer-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Swimmer-v3` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Walker2d-v2` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `Walker2d-v3` — ImportError: The mujoco v2 and v3 based environments have been moved to the gymnasium-robotics project (https://github.com/Farama-Foundation/gymnasium-robotics).
- `phys2d/CartPole-v0` — ModuleNotFoundError: No module named 'jax'
- `phys2d/CartPole-v1` — ModuleNotFoundError: No module named 'jax'
- `phys2d/Pendulum-v0` — ModuleNotFoundError: No module named 'jax'
- `tabular/Blackjack-v0` — ModuleNotFoundError: No module named 'jax'
- `tabular/CliffWalking-v0` — ModuleNotFoundError: No module named 'jax'
