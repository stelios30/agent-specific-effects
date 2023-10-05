from copy import deepcopy
from numpy.random import Generator
from typing import Any, Dict, List, Optional, Tuple

from ase.envs.causal import CausalEnv, Trajectory
from ase.actors.causal import CausalActor
from ase.actors.graph import GraphActor, GraphAction
from ase.tools.noise import NoiseModel, NullNoiseModel


class GraphTrajectory(Trajectory):
    def render(self) -> str:
        result = f"Trajectory: {self.id}\n"

        for t in range(self.horizon):
            result += f"    Time-Step: {t}\n"
            result += f"    State Id:  {self.states[t]['id']}\n"
            result += f"    State Step:  {self.states[t]['step']}\n"
            result += f"    Agent Levels: {self.states[t]['level']}\n"
            result += f"    Level Count: {self.states[t]['level_count']}\n"
            result += f"    Agent Actions: {self.actions[t]}\n"
            result += f"    Agent Probs: {self.probs[t]}\n"
            result += f"    Score: {self.states[t]['score']}\n"
            result += f"    Total Score: {self.states[t]['total_score']}\n\n"

        return result

    def failed(self) -> bool:
        return self.states[-1]["total_score"] <= 0.0


class GraphRewardModel:
    def reward(self, state: Dict[str, Any], act_taken: Tuple[int, ...], next_state: Dict[str, Any]) -> float:
        raise NotImplementedError


class BalancedLevelReward(GraphRewardModel):
    def reward(self, state: Dict[str, Any], act_taken: Tuple[int, ...], next_state: Dict[str, Any]) -> float:
        if next_state["id"] != "end":
            # intermediate reward is 0
            return 0.0
        elif len(set(next_state["level_count"])) == 1:
            # all levels have the same number of agents
            return +1.0
        else:
            # levels are not balanced
            return -1.0


class Graph(CausalEnv):
    def __init__(
        self,
        num_agents: int = 6,
        num_levels: int = 3,
        num_columns: int = 3,
        reward_constraint: GraphRewardModel = BalancedLevelReward(),
        act_noise_model: NoiseModel = NullNoiseModel(),
    ):
        super().__init__(act_noise_model=act_noise_model)
        assert num_levels % 2 == 1, "Number of levels must be odd."
        assert num_agents % num_levels == 0, "Number of agents must be divisible by number of levels."
        self.num_agents = num_agents
        self.num_columns = num_columns
        self.num_levels = num_levels
        self.reward_constraint = reward_constraint
        self.horizon = num_columns + 2

    def reset(self, rng: Optional[Generator] = None):
        state = {}
        state["id"] = "start"
        state["step"] = 0
        state["level"] = {agent_id: self.num_levels // 2 for agent_id in range(self.num_agents)}
        state["level_count"] = self._get_level_agent_count(state)
        state["score"] = 0.0
        state["total_score"] = 0.0
        return state

    def step(self, state: Dict[str, Any], actions: Tuple[int, ...], env_noise: Dict[str, List[NoiseModel]], rng: Optional[Generator] = None):
        next_state = deepcopy(state)

        # moving agents between levels
        for agent_id, act in enumerate(actions):
            if state["level"][agent_id] == 0 and act == GraphAction.up.value:
                # if agent attempts to move up on the highest level, it moves straight instead
                act = GraphAction.straight.value
            elif state["level"][agent_id] == self.num_levels - 1 and act == GraphAction.down.value:
                # if agent attempts to move down on the lowest level, it moves straight instead
                act = GraphAction.straight.value
            # moves the agent to the wanted level, c.f. GraphAction values
            next_state["level"][agent_id] += +1 if act == GraphAction.down else -1 if act == GraphAction.up else 0

        # determining successor state id
        if state["id"] == "start":
            next_state["id"] = 1
        elif state["id"] < self.horizon - 2:
            next_state["id"] = state["id"] + 1
        else:
            next_state["id"] = "end"

        # finalizing next state properties
        next_state["step"] += 1
        next_state["level_count"] = self._get_level_agent_count(next_state)
        next_state["score"] = self.reward_constraint.reward(state, actions, next_state)
        next_state["total_score"] += next_state["score"]

        return next_state

    def sample_trajectory(
        self,
        agents: List[GraphActor],
        act_noise: Optional[List[List[NoiseModel]]] = None,
        env_noise: Optional[List[Dict[str, List[NoiseModel]]]] = None,
        initial_state: Dict[str, Any] = None,
        do_operators: Optional[List[Dict[int, int]]] = None,
        rng: Optional[Generator] = None,
        horizon: Optional[int] = None,
        pad: bool = False,
    ) -> GraphTrajectory:
        t = super().sample_trajectory(agents, act_noise, env_noise, initial_state, do_operators, rng, horizon, pad)
        t = GraphTrajectory(id=t.id, states=t.states, actions=t.actions, probs=t.probs, env_noise=t.env_noise, act_noise=t.act_noise)
        return t

    def get_available_actions(self, state: Dict[str, Any], agent_id: int, strict: bool = False) -> List[int]:
        if state["id"] == "end":
            return []
        elif not strict:
            return [a.value for a in GraphAction]
        elif state["level"][agent_id] == 0:
            return [GraphAction.down.value, GraphAction.straight.value]
        elif state["level"][agent_id] == self.num_levels - 1:
            return [GraphAction.up.value, GraphAction.straight.value]
        else:
            return [a.value for a in GraphAction]

    def _get_acting_agents(self, state: Dict[str, Any], agents: List[CausalActor]) -> List[CausalActor]:
        return agents

    def _get_level_agent_count(self, state: Dict[str, Any]) -> List[int]:
        result = [0] * self.num_levels
        for agent_level in state["level"].values(): result[agent_level] += 1
        return result
