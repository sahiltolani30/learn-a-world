"""
Agent Belief & Recurrent State Tracker for Learn-a-World v3 (R2).
Maintains both:
1. Unlocked achievement mask m in {0, 1}^22
2. Recurrent memory state h in R^256 (temporal context for nights, occlusions, and long chains)
WITHOUT accessing ground-truth environment info.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from eval.metrics import CRAFTER_ACHIEVEMENTS

NUM_ACHIEVEMENTS = len(CRAFTER_ACHIEVEMENTS)  # 22


class AgentBeliefV3:
    """
    Epistemic & Recurrent state tracker for v3 planning.
    Tracks unlocked achievements and temporal hidden state strictly from transitions.
    """

    def __init__(
        self,
        recurrent_dim: int = 256,
        frontier_threshold: float = 0.25,
        common_threshold: float = 0.50,
        event_thresholds: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
        strict_inference: bool = True,
    ):
        self.recurrent_dim = recurrent_dim
        self.strict_inference = strict_inference
        self.m_bin = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
        self.m_soft = np.zeros(NUM_ACHIEVEMENTS, dtype=np.float32)
        self.h = np.zeros(recurrent_dim, dtype=np.float32)
        self.history = []

        # Adaptive Frontier Thresholds (per achievement)
        if event_thresholds is not None:
            if isinstance(event_thresholds, (int, float)):
                self.thresholds = np.full(NUM_ACHIEVEMENTS, float(event_thresholds), dtype=np.float32)
            elif isinstance(event_thresholds, torch.Tensor):
                self.thresholds = event_thresholds.detach().cpu().numpy().astype(np.float32).reshape(-1)
            else:
                self.thresholds = np.array(event_thresholds, dtype=np.float32).reshape(-1)
        else:
            if self.strict_inference:
                # In strict mode: uniform calibrated threshold across all achievements.
                # Never assign hardcoded Crafter achievement indices special meaning!
                self.thresholds = np.full(NUM_ACHIEVEMENTS, frontier_threshold, dtype=np.float32)
            else:
                # Privileged / diagnostic mode ONLY:
                self.thresholds = np.full(NUM_ACHIEVEMENTS, common_threshold, dtype=np.float32)
                frontier_indices = [0, 1, 3, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 19, 20]
                self.thresholds[frontier_indices] = frontier_threshold

        # Cognitive Working Memory (System 1 - Within-Episode Context & Landmark Anchors)
        from planning.semantic_memory import ShortTermWorkingMemory, inventory_dict_to_vector
        self.working_memory = ShortTermWorkingMemory(capacity=16, recurrent_dim=recurrent_dim)
        self.inv_vec = inventory_dict_to_vector(None)

    @property
    def mask(self) -> np.ndarray:
        """Binary mask property for backward compatibility."""
        return self.m_bin

    @mask.setter
    def mask(self, val: np.ndarray) -> None:
        self.m_bin = np.array(val, dtype=np.float32)

    def reset(self) -> None:
        """Resets belief masks, recurrent hidden state, and working memory at episode start."""
        self.m_bin.fill(0.0)
        self.m_soft.fill(0.0)
        self.h.fill(0.0)
        self.history.clear()
        self.working_memory.reset()
        from planning.semantic_memory import inventory_dict_to_vector
        self.inv_vec = inventory_dict_to_vector(None)

    def update_inventory(self, inv: Union[Dict[str, Any], np.ndarray, torch.Tensor]) -> None:
        """Updates internal 16-D backpack vector from environment inventory info."""
        if self.strict_inference:
            raise RuntimeError(
                "Updating inventory from environment info is forbidden in strict inference mode. "
                "Set strict_inference=False only for privileged diagnostic runs."
            )
        if isinstance(inv, dict):
            from planning.semantic_memory import inventory_dict_to_vector
            self.inv_vec = inventory_dict_to_vector(inv)
        elif isinstance(inv, torch.Tensor):
            self.inv_vec = inv.detach().cpu().numpy().astype(np.float32).reshape(-1)
        elif isinstance(inv, np.ndarray):
            self.inv_vec = inv.astype(np.float32).reshape(-1)

    def get_inv_vector(self) -> np.ndarray:
        """Returns copy of current 16-D backpack vector."""
        return self.inv_vec.copy()

    def get_inv_tensor(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Returns current 16-D backpack vector as PyTorch tensor."""
        t = torch.from_numpy(self.inv_vec).float()
        if device is not None:
            t = t.to(device)
        return t

    def update_working_memory(
        self,
        z_t: torch.Tensor,
        action: int,
        reward: float,
        next_h: torch.Tensor,
        inventory_dict: Optional[Dict[str, Any]] = None,
        unlocked_event: Optional[str] = None,
        dopamine_surge: float = 0.0,
    ) -> None:
        """Explicit hook to update short-term working memory with latent z_t, backpack state, and landmark anchors."""
        if inventory_dict is not None:
            self.update_inventory(inventory_dict)
        self.working_memory.update(
            z_t=z_t,
            action=action,
            reward=reward,
            next_h=next_h,
            inventory_mask=self.m_bin,
            inventory_dict=inventory_dict,
            unlocked_event=unlocked_event,
            dopamine_surge=dopamine_surge,
        )

    def update(
        self,
        event_probs: Union[np.ndarray, torch.Tensor],
        reward: float,
        next_h: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """
        Updates belief masks and advances recurrent hidden state:
        event_probs: [22] probabilities from EventHead
        reward: scalar reward received from environment
        next_h: [recurrent_dim] updated hidden state from RecurrentCore
        """
        if isinstance(event_probs, torch.Tensor):
            p = event_probs.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            p = np.array(event_probs, dtype=np.float32).reshape(-1)

        assert len(p) == NUM_ACHIEVEMENTS, f"Expected {NUM_ACHIEVEMENTS} event probs, got {len(p)}"

        if next_h is not None:
            if isinstance(next_h, torch.Tensor):
                self.h = next_h.detach().cpu().numpy().astype(np.float32).reshape(-1)
            else:
                self.h = np.array(next_h, dtype=np.float32).reshape(-1)

        # 1. Soft mask: running maximum of predicted probability (used for reward)
        self.m_soft = np.maximum(self.m_soft, p)

        # 2. Binary mask: thresholded with adaptive frontier thresholds
        locked_indices = np.where(self.m_bin < 0.5)[0]
        newly_unlocked = []

        if len(locked_indices) > 0:
            for idx in locked_indices:
                if p[idx] > self.thresholds[idx]:
                    self.m_bin[idx] = 1.0
                    newly_unlocked.append(idx)

            # 3. Fallback rule: reward fired (>= 0.9) but no event probability exceeded threshold
            # Pins BOTH masks to 1.0 for that head as ground truth from the environment
            if float(reward) >= 0.9 and len(newly_unlocked) == 0:
                locked_probs = p[locked_indices]
                best_locked_idx = locked_indices[int(np.argmax(locked_probs))]
                self.m_bin[best_locked_idx] = 1.0
                self.m_soft[best_locked_idx] = 1.0
                newly_unlocked.append(best_locked_idx)

        self.history.append({
            "event_probs": p.copy(),
            "reward": float(reward),
            "newly_unlocked": newly_unlocked,
            "m_bin": self.m_bin.copy(),
            "m_soft": self.m_soft.copy(),
            "mask": self.m_bin.copy(),
        })

    def set_mask(
        self,
        mask: Union[np.ndarray, torch.Tensor],
        soft_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        """Sets belief masks directly (used for true mask updates or snapshot restoration)."""
        if isinstance(mask, torch.Tensor):
            m = mask.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            m = np.array(mask, dtype=np.float32).reshape(-1)
        assert len(m) == NUM_ACHIEVEMENTS, f"Expected {NUM_ACHIEVEMENTS} elements, got {len(m)}"
        self.m_bin = m.copy()

        # Keep 16-D backpack vector synchronized with mask updates
        from planning.semantic_memory import CRAFTER_INVENTORY_KEYS
        for idx, ach_name in enumerate(CRAFTER_ACHIEVEMENTS):
            if m[idx] > 0.5:
                for inv_key in CRAFTER_INVENTORY_KEYS[:12]:
                    if ach_name == f"collect_{inv_key}" or ach_name == f"make_{inv_key}":
                        inv_idx = CRAFTER_INVENTORY_KEYS.index(inv_key)
                        self.inv_vec[inv_idx] = max(self.inv_vec[inv_idx], 1.0)

        if soft_mask is not None:
            if isinstance(soft_mask, torch.Tensor):
                s_arr = soft_mask.detach().cpu().numpy().astype(np.float32).reshape(-1)
            else:
                s_arr = np.array(soft_mask, dtype=np.float32).reshape(-1)
            assert len(s_arr) == NUM_ACHIEVEMENTS, f"Expected {NUM_ACHIEVEMENTS} elements, got {len(s_arr)}"
            self.m_soft = s_arr.copy()
        else:
            self.m_soft = np.maximum(self.m_soft, m)

    def set_h(self, h: Union[np.ndarray, torch.Tensor]) -> None:
        """Sets recurrent memory state h directly (used for snapshot restoration)."""
        if isinstance(h, torch.Tensor):
            h_arr = h.detach().cpu().numpy().astype(np.float32).reshape(-1)
        else:
            h_arr = np.array(h, dtype=np.float32).reshape(-1)
        assert len(h_arr) == self.recurrent_dim, f"Expected {self.recurrent_dim} elements, got {len(h_arr)}"
        self.h = h_arr.copy()

    def get_mask(self) -> np.ndarray:
        """Returns copy of binary unlocked mask [22] (for model input)."""
        return self.m_bin.copy()

    def get_mask_tensor(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Returns binary mask as PyTorch tensor (for model input)."""
        t = torch.from_numpy(self.m_bin).float()
        if device is not None:
            t = t.to(device)
        return t

    def get_soft_mask(self) -> np.ndarray:
        """Returns copy of soft belief mask [22] (running maximum for reward)."""
        return self.m_soft.copy()

    def get_soft_mask_tensor(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Returns soft mask as PyTorch tensor (for reward / TTE)."""
        t = torch.from_numpy(self.m_soft).float()
        if device is not None:
            t = t.to(device)
        return t

    def get_binary_mask(self) -> np.ndarray:
        """Alias for get_mask()."""
        return self.m_bin.copy()

    def get_binary_mask_tensor(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Alias for get_mask_tensor()."""
        return self.get_mask_tensor(device=device)

    def get_h(self) -> np.ndarray:
        """Returns copy of recurrent hidden state [recurrent_dim]."""
        return self.h.copy()

    def get_h_tensor(self, device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
        """Returns recurrent hidden state as PyTorch tensor."""
        t = torch.from_numpy(self.h).float()
        if device is not None:
            t = t.to(device)
        return t
