import numpy as np

from typing import Dict, List, Optional

from ase.envs.causal import Trajectory, CausalEnv, CausalActor


def tcfe(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, num_cf_samples: int = 100, seed: Optional[int] = None):
    """Implements calculation of the total counterfactual effect.
    Args:
        trajectory (Trajectory): The trajectory to calculate the total counterfactual effect for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual calculation. Defaults to 100.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The total counterfactual effect of the intervention.
    """
    num_successes = 0

    for i in range(num_cf_samples):
        rng = np.random.default_rng(seed + i)

        # sample noise from posterior
        act_noise, env_noise = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        # construct the intervention
        do_operators = [{} for _ in range(trajectory.horizon)]
        do_operators[time_step] = intervention

        # sample counterfactual outcome
        outcome = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        initial_state=trajectory.states[0], horizon=trajectory.horizon, rng=rng)
        num_successes += outcome.success()

    return num_successes / num_cf_samples


def pse(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], time_step: int, num_cf_samples: int = 100, seed: Optional[int] = None):
    """Implements calculation of the path-specific effect.
    Args:
        trajectory (Trajectory): The trajectory to calculate the total counterfactual effect for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        time_step (int): The time-step of the intervention.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual calculation. Defaults to 100.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The path-specific effect of the intervention.
    """
    num_successes = 0

    for i in range(num_cf_samples):
        rng = np.random.default_rng(seed + i)

        # sample noise from posterior
        act_noise, env_noise = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        # construct the intervention
        do_agents = list(intervention.keys())
        do_operators = [{} for _ in range(trajectory.horizon)]
        do_operators[time_step] = intervention
        for t in range(time_step + 1, trajectory.horizon):
            do_operators[t] = {agent_id: trajectory.actions[t][agent_id] for agent_id in do_agents}

        # sample counterfactual outcome
        outcome = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        rng=rng, initial_state=trajectory.states[0], horizon=trajectory.horizon)
        num_successes += outcome.success()

    return num_successes / num_cf_samples


def ase(trajectory: Trajectory, env: CausalEnv, agents: List[CausalActor], intervention: Dict[int, int], effect_agents: List[int], time_step: int, num_cf_samples: int = 100, seed: Optional[int] = None):
    """Implements calculation of the agent-specific effect.
    Args:
        trajectory (Trajectory): The trajectory to calculate the total counterfactual effect for.
        env (CausalEnv): Instance of the environment to sample counterfactuals from.
        agents (List[CausalActor]): List of agents acting in the environment.
        intervention (Dict[int, int]): The intervention to apply, with keys indicating agent's id and values indicating agent's action.
        effect_agents: (List[int]): List of agent ids to calculate the effect for.
        time_step (int): The time-step of the intervention.
        num_cf_samples (int, optional): Number of posterior samples to use for counterfactual calculation. Defaults to 100.
        seed (Optional[int], optional): Optional seed used for reproducibility. Defaults to None.
    Returns:
        int: The path-specific effect of the intervention.
    """
    assert 1 == len(intervention.keys()), "ASE only supports single agent interventions."
    num_successes = 0

    for i in range(num_cf_samples):
        rng = np.random.default_rng(seed + i)

        # sample noise from posterior
        act_noise, env_noise = env.sample_noise_from_posterior(trajectory=trajectory, agents=agents, rng=rng)

        # sample counterfactual trajectory
        do_operators = [{} for _ in range(trajectory.horizon)]
        do_operators[time_step] = intervention
        traj_cf = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        initial_state=trajectory.states[0], rng=rng, pad=True, horizon=trajectory.horizon)

        # construct the intervention
        do_operators = [{} for _ in range(trajectory.horizon)]
        non_effect_agents = [agent.id for agent in agents if agent.id not in effect_agents]

        for t in range(time_step + 1, trajectory.horizon):
            for agent_id in non_effect_agents:
                do_operators[t][agent_id] = trajectory.actions[t][agent_id]
            for agent_id in effect_agents:
                do_operators[t][agent_id] = traj_cf.actions[t][agent_id]

        # sample counterfactual outcome
        outcome = env.sample_trajectory(agents, act_noise=act_noise, env_noise=env_noise, do_operators=do_operators,
                                        initial_state=trajectory.states[0], rng=rng, horizon=trajectory.horizon)
        num_successes += outcome.success()

    return num_successes / num_cf_samples
