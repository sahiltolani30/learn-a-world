"""Update scheduler coordinating environment steps and learner updates."""


class UpdateScheduler:
    """Calculates when optimizer updates should be triggered based on updates_per_env_step."""

    def __init__(self, updates_per_env_step: float = 0.25):
        self.updates_per_env_step = updates_per_env_step
        self.env_steps = 0
        self.learner_updates = 0
        self._accumulator = 0.0

    def step(self, num_env_steps: int = 1) -> int:
        """Register completed environment steps and return the number of learner updates to perform."""
        self.env_steps += num_env_steps
        self._accumulator += num_env_steps * self.updates_per_env_step
        
        num_updates = int(self._accumulator)
        self._accumulator -= num_updates
        self.learner_updates += num_updates
        return num_updates

    def get_state(self) -> dict:
        return {
            "updates_per_env_step": self.updates_per_env_step,
            "env_steps": self.env_steps,
            "learner_updates": self.learner_updates,
            "accumulator": self._accumulator,
        }

    def set_state(self, state: dict) -> None:
        self.updates_per_env_step = state["updates_per_env_step"]
        self.env_steps = state["env_steps"]
        self.learner_updates = state["learner_updates"]
        self._accumulator = state["accumulator"]
