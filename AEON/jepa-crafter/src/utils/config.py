"""Configuration loader, dataclasses, and validator for JEPA-Crafter."""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class ProjectConfig:
    name: str = "jepa_crafter"
    seed: int = 12345
    device: str = "auto"
    output_dir: str = "runs/"
    checkpoint_every_env_steps: int = 10000


@dataclass
class EnvironmentConfig:
    package: str = "crafter"
    env_id: str = "CrafterReward-v1"
    observation_shape: List[int] = field(default_factory=lambda: [64, 64, 3])
    action_count: int = 17
    opaque_action_permutation: str = "random_per_run"
    reward_mode: str = "scalar"
    max_environment_steps: int = 1000000


@dataclass
class CollectionConfig:
    initial_random_steps: int = 10000
    num_env_workers: int = 1
    train_seed_start: int = 1000
    heldout_seed_start: int = 900000
    heldout_episode_count: int = 50


@dataclass
class ReplayConfig:
    capacity: int = 50000
    observation_dtype: str = "uint8"
    recent_sequence_fraction: float = 0.30
    uniform_sequence_fraction: float = 0.70
    context_length: int = 4
    prediction_horizon: int = 8
    min_sequences_before_training: int = 2000


@dataclass
class ModelConfig:
    encoder_channels: List[int] = field(default_factory=lambda: [64, 96, 128])
    latent_tokens: int = 64
    latent_dim: int = 128
    action_embedding_dim: int = 64
    context_hidden_dim: int = 256
    context_length: int = 4
    predictor_members: int = 3
    predictor_transformer_blocks: int = 2
    transformer_heads: int = 4
    transformer_mlp_dim: int = 512
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    target_mask_fraction: float = 0.50
    ema_momentum_start: float = 0.996
    ema_momentum_end: float = 0.999
    ema_momentum_ramp_updates: int = 50000


@dataclass
class OptimizationConfig:
    optimizer: str = "AdamW"
    encoder_lr: float = 0.0001
    predictor_lr: float = 0.0003
    auxiliary_lr: float = 0.0003
    weight_decay: float = 0.0001
    batch_size: int = 32
    gradient_clip_norm: float = 1.0
    warmup_updates: int = 1000
    precision: str = "fp16"
    updates_per_env_step: float = 0.25
    reward_loss_weight: float = 0.10
    termination_loss_weight: float = 0.05


@dataclass
class ExplorationConfig:
    novelty_weight: float = 1.0
    disagreement_weight: float = 0.25
    learning_progress_weight: float = 0.25
    random_action_floor: float = 0.10
    intrinsic_clip: float = 5.0


@dataclass
class PlannerConfig:
    enabled_after_validation: bool = True
    algorithm: str = "cem"
    horizon: int = 8
    population: int = 64
    iterations: int = 2
    elite_count: int = 8
    discount: float = 0.99
    replan_every_real_step: bool = True
    execute_first_action_only: bool = True
    max_intrinsic_bonus_per_step: float = 2.0


@dataclass
class PopulationConfig:
    enabled_after_single_role_validation: bool = True
    roles: List[str] = field(default_factory=lambda: ["explorer", "scientist", "experimenter", "balanced_planner"])
    shared_world_model: bool = True
    shared_replay: bool = True


@dataclass
class EvaluationConfig:
    num_development_seeds: int = 3
    num_final_seeds: int = 5
    report_all_22_achievements: bool = True
    heldout_prediction_horizons: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    evaluate_mpc_ablation: bool = True
    save_videos: bool = True


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return cls(
            project=ProjectConfig(**d.get("project", {})),
            environment=EnvironmentConfig(**d.get("environment", {})),
            collection=CollectionConfig(**d.get("collection", {})),
            replay=ReplayConfig(**d.get("replay", {})),
            model=ModelConfig(**d.get("model", {})),
            optimization=OptimizationConfig(**d.get("optimization", {})),
            exploration=ExplorationConfig(**d.get("exploration", {})),
            planner=PlannerConfig(**d.get("planner", {})),
            population=PopulationConfig(**d.get("population", {})),
            evaluation=EvaluationConfig(**d.get("evaluation", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


def load_config(path: str) -> Config:
    """Load config from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config.from_dict(data)


def save_config(config: Config, path: str) -> None:
    """Save config to a YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
