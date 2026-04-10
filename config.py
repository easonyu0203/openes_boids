from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # World
    n_agents: int = 32
    box_size: float = 10.0          # Square box [0, box_size]^2

    # Agent physics
    agent_speed: float = 2.0        # Constant speed (units/s)
    turn_rate: float = 2.0          # Angular velocity when turning (rad/s)
    collision_radius: float = 0.3   # Agents closer than this collide

    # Goal
    goal_radius: float = 1.0        # Radius of the food circle
    goal_speed: float = 1.5         # Speed of the goal random walk

    # Observation
    n_obs_agents: int = 6           # K closest other agents to observe

    # Simulation
    fps: int = 30
    episode_len: int = 300          # Steps per trajectory

    # Rewards
    collision_penalty: float = -1.0
    wall_penalty: float = -0.5
    goal_reward: float = 1.0

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def feat_dim(self) -> int:
        # Per token: [vx, vy, omega, d_left, d_right, d_bottom, d_top,
        #             dist_to_goal, angle_to_goal, dist_to_ego, angle_to_ego]
        return 11

    @property
    def obs_shape(self) -> tuple[int, int]:
        # tokens: 1 self + K neighbours + 1 goal = K+2
        return (self.n_obs_agents + 2, self.feat_dim)


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_blocks: int = 2
    n_actions: int = 3             # turn-left, no-turn, turn-right


@dataclass
class ESConfig:
    pop_size: int = 256             # Must be even (antithetic pairs)
    noise_std: float = 0.1
    lr: float = 0.01
    n_generations: int = 2000
    noise_table_size: int = 25_000

    # Logging & checkpoints
    log_every: int = 10
    checkpoint_every: int = 100
    checkpoint_dir: str = "checkpoints"
    log_file: str = "training_log.csv"

    def __post_init__(self):
        assert self.pop_size % 2 == 0, "pop_size must be even for antithetic sampling"
