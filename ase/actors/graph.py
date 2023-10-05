from enum import Enum
from typing import Any, Dict, List, Union, Tuple, Optional

from ase.tools.noise import NoiseModel

from .causal import CausalActor


class GraphAction(int, Enum):
    up = 0
    down = 1
    straight = 2


class GraphActor(CausalActor):
    def __init__(self, id: int, p_i: Optional[float] = None):
        super().__init__(id)
        self.p_i = 0.05 * (self.id + 1) if p_i is None else p_i

    def policy(self, state: Dict[str, Any]) -> List[float]:
        obs = self._get_obs(state)

        n_k = obs["level_count"][obs["level"]]
        n_act = len(GraphAction)
        p_rand = 1 / n_act

        if obs["id"] == "start":
            # in the initial state, agent selects action uniform-random
            p_up = p_down = p_straight = p_rand

        if n_k <= 2:
            # number of agents Nk is <= 2, agent goes straight with probability 1 - p_i
            p_straight = self.p_i * p_rand + (1 - self.p_i)
            p_up, p_down = self.p_i * p_rand, self.p_i * p_rand

        if n_k > 2:
            # agent takes action `straight` with probability (1 - p_i) * 2 / Nk
            p_straight = self.p_i * p_rand + ((1 - self.p_i) * 2 / n_k)
            p_up, p_down = self.p_i * p_rand, self.p_i * p_rand

            # agent goes towards row occupied by less than 2 agents with probability (1 - pi) * (n_k - 2) / n_k
            target_level = [level for level in range(len(obs["level_count"])) if obs["level_count"][level] < 2]
            assert len(target_level) > 0, "There should be at least one row with less than 2 agents."

            if len(target_level) == 1 and target_level[0] < obs["level"]:
                # agent goes up if target row is above its current row
                p_up += (1 - self.p_i) * (n_k - 2) / n_k
            elif len(target_level) == 1 and target_level[0] > obs["level"]:
                # agents goes down if target row is below its current row
                p_down += (1 - self.p_i) * (n_k - 2) / n_k
            elif len(target_level) > 1:
                # there are multiple rows with less than 2 agents, ties are broken uniform-random
                p_up += 1/2 * ((1 - self.p_i) * (n_k - 2) / n_k)
                p_down += 1/2 * ((1 - self.p_i) * (n_k - 2) / n_k)
            else:
                raise RuntimeError("We should not be here.")

        p = [p_up, p_down, p_straight]
        assert sum(p) - 1 < 1e-5, f"Probabilities should sum to 1. Observed: {sum([p_up, p_down, p_straight])}"
        assert all(p) > 0, f"Probabilities should be positive."
        return p

    def action(self, state: Dict[str, Any], act_noise: NoiseModel, return_probs: bool = False) -> Union[int, Tuple[int, List[float]]]:
        # obtains agent's policy probabilities
        probs = self.policy(state)
        acts = [GraphAction.up.value, GraphAction.down.value, GraphAction.straight.value]

        # samples from agent's policy, with noise
        action = act_noise.choice(probs)
        action = acts[action]

        return action if not return_probs else (action, probs)

    def _get_obs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": state["id"],
            "level": state["level"][self.id],
            "level_count": state["level_count"],
        }
