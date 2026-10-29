"""
Dual-Drive Homeostatic-Teleological Arbiter for Godfather JEPA.

Reconciles the fundamental conflict between:
1. Homeostatic Drive (Survival): Regulating thirst, hunger, health, and nightfall danger.
2. Teleological Drive (Tech-Tree): Advancing tool crafting and exploring new frontiers.

Prevents the fatal flaw where agents dehydrated to 1 HP while collecting wood,
dying instantly on the first zombie strike at nightfall step 150.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


class HomeostaticArbiter:
    """
    Arbitrates between survival maintenance reflexes and goal-directed tech-tree execution.
    """

    def __init__(
        self,
        thirst_threshold: float = 3.0,
        hunger_threshold: float = 3.0,
        health_threshold: float = 4.0,
        dusk_step_threshold: Optional[int] = None,
        night_end_step: Optional[int] = None,
        combat_cadence: int = 3,
        causal_hypergraph: Optional[Any] = None,
    ):
        self.thirst_threshold = thirst_threshold
        self.hunger_threshold = hunger_threshold
        self.health_threshold = health_threshold
        self.dusk_step_threshold = dusk_step_threshold
        self.night_end_step = night_end_step
        self.combat_cadence = combat_cadence
        self.causal_hypergraph = causal_hypergraph

    def get_action(self, card_name: str, fallback: int = 5) -> int:
        """Dynamically retrieves action for given vital achievement from learned causal cards."""
        if self.causal_hypergraph is not None and hasattr(self.causal_hypergraph, "cards"):
            card = self.causal_hypergraph.cards.get(card_name)
            if card is not None:
                return card.action
        return fallback

    def evaluate_urgency(
        self,
        vitals: Dict[str, float],
        step: int = 0,
        continuation_prob: float = 1.0,
        visual_darkness: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Computes continuous urgency signals across vital dimensions.
        When any vital drops to or below its maintenance threshold, its urgency
        is calibrated to fall in [0.5, 1.0], guaranteeing homeostatic priority.
        Rely on learned continuation probability c(s_t) and visual signals instead
        of hardcoded environmental clock constants.
        """
        drink = vitals.get("drink", 9.0)
        food = vitals.get("food", 9.0)
        health = vitals.get("health", 9.0)

        # Calibrate u_thirst: in [0.5, 1.0] when drink <= thirst_threshold
        if drink <= self.thirst_threshold:
            u_thirst = 0.5 + 0.5 * max(0.0, (self.thirst_threshold - drink) / max(self.thirst_threshold, 1e-6))
        else:
            u_thirst = 0.0

        # Calibrate u_hunger: in [0.5, 1.0] when food <= hunger_threshold
        if food <= self.hunger_threshold:
            u_hunger = 0.5 + 0.5 * max(0.0, (self.hunger_threshold - food) / max(self.hunger_threshold, 1e-6))
        else:
            u_hunger = 0.0

        # Calibrate u_health: in [0.5, 1.0] when health <= health_threshold
        if health <= self.health_threshold:
            u_health = 0.5 + 0.5 * max(0.0, (self.health_threshold - health) / max(self.health_threshold, 1e-6))
        else:
            u_health = 0.0

        # Ambient darkness factor: from visual input or continuation head drop
        if visual_darkness is not None:
            dusk_factor = float(visual_darkness)
        elif self.dusk_step_threshold is not None and self.night_end_step is not None:
            cycle_step = step % 300
            dusk_factor = 0.5 if (self.dusk_step_threshold <= cycle_step <= self.night_end_step) else 0.0
        else:
            dusk_factor = 0.5 if continuation_prob <= 0.3 else 0.0

        # Imminent combat threat from survival continuation head drop (predicted death)
        imminent_threat = max(0.0, 1.0 - continuation_prob)
        u_danger = max(dusk_factor, imminent_threat)

        total_urgency = max(u_thirst, u_hunger, u_health, u_danger)

        return {
            "total_urgency": total_urgency,
            "u_thirst": u_thirst,
            "u_hunger": u_hunger,
            "u_health": u_health,
            "u_danger": u_danger,
            "dusk_factor": dusk_factor,
            "imminent_threat": imminent_threat,
        }

    def arbitrate(
        self,
        vitals: Dict[str, float],
        step: int,
        teleological_action: int,
        water_adjacent: bool = False,
        food_adjacent: bool = True,
        monster_adjacent: bool = False,
        can_sleep: bool = False,
        continuation_prob: float = 1.0,
        visual_darkness: Optional[float] = None,
    ) -> Tuple[int, str]:
        """
        Selects whether to execute the tech-tree action or override with a homeostatic reflex.

        Priority Ordering:
        1. Preemptive Drinking: Opportunistic Action 5 if adjacent to water and thirst <= threshold.
        2. Imminent Combat Defense: Action 5 if monster is adjacent or continuation_prob <= 0.2.
        3. Nightfall Sleep Reflex: Action 6 to safely skip night if sleep conditions met.
        4. Safe: Teleological Drive active when total urgency < 0.5.
        5. Urgent Hunger Override: Action 5 (forage/eat) when hunger urgency >= 0.5.
        6. Modulated Dusk Maneuver: Striking on cadence intervals while allowing movement on others.
        7. Teleological Fallback: Default to tech-tree action if maintenance preconditions missing.

        Returns:
            (selected_action, active_drive_mode)
        """
        drink = vitals.get("drink", 9.0)
        urgency = self.evaluate_urgency(
            vitals, step, continuation_prob=continuation_prob, visual_darkness=visual_darkness
        )

        # 1. Preemptive opportunistic drinking reflex (Highest vital priority)
        # Triggers whenever adjacent to water and thirst is at or below threshold
        if water_adjacent and (drink <= self.thirst_threshold or urgency["u_thirst"] >= 0.5):
            return self.get_action("collect_drink", fallback=5), "homeostatic_drink"

        # 2. Imminent combat defense (Active monster attack in progress)
        if urgency["imminent_threat"] >= 0.8 or monster_adjacent:
            return self.get_action("defeat_zombie", fallback=5), "homeostatic_combat"

        # 3. Nightfall sleep reflex (Skip dangerous night hours if possible)
        is_night = (urgency["dusk_factor"] >= 0.5) or (urgency["imminent_threat"] >= 0.5)
        if can_sleep and is_night:
            return self.get_action("sleep", fallback=6), "homeostatic_sleep"

        # 4. Safe: Teleological Drive active
        if urgency["total_urgency"] < 0.5:
            return teleological_action, "teleological"

        # 5. Urgent hunger override (forage / eat plant or cow)
        if urgency["u_hunger"] >= 0.5 and food_adjacent:
            return self.get_action("eat_cow", fallback=5), "homeostatic_eat"

        # 6. Active inference survival defense (zero artificial clock cadence modulos)
        # When dusk/darkness is present, defend if threats emerge or continue tactical maneuvering
        if urgency["dusk_factor"] >= 0.5:
            if urgency["imminent_threat"] >= 0.6 or monster_adjacent:
                return self.get_action("defeat_zombie", fallback=5), "homeostatic_combat"
            return teleological_action, "teleological_dusk_maneuver"

        # 7. Default fallback to teleological action if maintenance precondition unavailable
        return teleological_action, "teleological_fallback"
