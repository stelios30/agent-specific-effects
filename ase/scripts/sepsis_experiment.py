import time
import tqdm
import typer
import pickle
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp

from joblib import Parallel, delayed
from numpy.random import Generator
from pathlib import Path
from typing import List, Optional, Tuple
from rich import print
from typing_extensions import Annotated

from ase.envs.sepsis import Sepsis, SepsisAction, State, SepsisTrajectory
from ase.actors.sepsis import ClinicianActor, AIActor
from ase.tools.noise import UniformNoiseModel, SepsisStateUniformNoiseModel
from ase.tools.order import Order
from ase.tools.algorithms import tcfe, pse, ase
from ase.tools.utils import find_by_id, sample_trajectories


def main(
    seeds: Annotated[List[int], typer.Argument(help="List of random seeds to use for repeated experiment runs.")],
    artifacts_dir: Annotated[Path, typer.Option(help="Path to directory where artifacts are stored. There will be one subdirectory per random seed.", dir_okay=True, file_okay=False)],
    mdp_path: Annotated[Path, typer.Option(help="Path to exported MDP dynamics (see `learn_sepsis_mdp.ipynb` notebook).", dir_okay=False, file_okay=True)],
    cl_policy_path: Annotated[Path, typer.Option(help="Path to exported clinician policy (see `learn_sepsis_actors.ipynb` notebook).", dir_okay=False, file_okay=True)],
    ai_policy_path: Annotated[Path, typer.Option(help="Path to exported AI policy (see `learn_sepsis_actors.ipynb` notebook).", dir_okay=False, file_okay=True)],
    tcfe_threshold: Annotated[float, typer.Option(help="Minimum TCFE an intervention needs to have to be considered for analysis.")] = 0.8,
    posterior_sample_complexity: Annotated[Optional[int], typer.Option(help="Run the posterior sample complexity analysis for given number of counterfactuals.")] = None,
    shuffle_total_order: Annotated[int, typer.Option(help="Run the robustness analysis w.r.t. misspecified state and action total order.")] = 0,
    num_trajectories: Annotated[int, typer.Option(help="Number of trajectories to sample for analysis, per trust level.")] = 100,
    num_cf_samples: Annotated[int, typer.Option(help="Number of counterfactual samples to draw when calculating ASE/TCFE.")] = 100,
    trust_values: Annotated[str, typer.Option(help="Comma-separated list of trust values to use for clinician's policy.")] = "0.0,0.25,0.5,0.75,1.0",
    max_horizon: Annotated[int, typer.Option(help="Maximum horizon of sampled trajectories.")] = 20,
):
    time_start = time.time()

    for seed in seeds:
        # sets up training artifacts and seed
        artifacts = artifacts_dir / str(seed)
        artifacts.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        trust_levels = [float(t) for t in trust_values.split(",")]
        print(f"Running experiment for seed {seed}.")

        # loads or samples trajectories and counterfactuals, for each trust level
        trajectories, counterfactuals = _enumerate_or_load_counterfactuals(artifacts_dir=artifacts, num_trajectories=num_trajectories, trust_levels=trust_levels,
                                                                           ai_policy_path=ai_policy_path, cl_policy_path=cl_policy_path, mdp_path=mdp_path, rng=rng,
                                                                           max_horizon=max_horizon, num_cf_samples=num_cf_samples)

        # runs posterior sample complexity analysis, if requested
        if posterior_sample_complexity:
            _ = _calculate_or_load_posterior_sample_complexity_analysis(artifacts_dir=artifacts, counterfactuals=counterfactuals.sample(posterior_sample_complexity, random_state=rng),
                                                                        trajectories=trajectories, rng=rng, mdp_path=mdp_path, ai_policy_path=ai_policy_path, cl_policy_path=cl_policy_path,
                                                                        max_horizon=max_horizon)

        # select only those interventions with specified TCFE and whose actions were taken at least two time steps before the end of the trajectory
        counterfactuals = counterfactuals.loc[np.where((counterfactuals.tcfe >= tcfe_threshold))]
        print(f"Selected {len(counterfactuals)} counterfactuals with TCFE >= {tcfe_threshold} for analysis.")

        # calculate ASE and PSE for all selected interventions
        counterfactuals = _calculate_causal_quantities(artifacts_dir=artifacts, counterfactuals=counterfactuals, trajectories=trajectories, rng=rng, num_cf_samples=num_cf_samples,
                                                       mdp_path=mdp_path, ai_policy_path=ai_policy_path, cl_policy_path=cl_policy_path, max_horizon=max_horizon)

        # if requested, calculate ASE for misspecified total order
        if shuffle_total_order:
            _ = _calculate_causal_quantities_with_misspecified_order(artifacts_dir=artifacts, counterfactuals=counterfactuals, trajectories=trajectories, rng=rng, num_cf_samples=num_cf_samples,
                                                                     mdp_path=mdp_path, ai_policy_path=ai_policy_path, cl_policy_path=cl_policy_path, max_horizon=max_horizon, num_order_shuffles=shuffle_total_order)

    print(f"Experiment finished. Time elapsed: {time.time() - time_start} seconds.")


def _enumerate_or_load_counterfactuals(
    artifacts_dir: Path, num_trajectories: int, trust_levels: List[float],
    ai_policy_path: str, cl_policy_path: str, mdp_path: str,
    max_horizon: int, num_cf_samples: int, rng: Generator,
) -> Tuple[List[SepsisTrajectory], pd.DataFrame]:
    trajectories_pkl_path = artifacts_dir / "trajectories.pkl"
    trajectories_txt_path = artifacts_dir / "trajectories.txt"
    counterfactuals_path = artifacts_dir / "counterfactuals.csv"

    # if we have already sampled trajectories and counterfactuals, we load them from disk
    if trajectories_pkl_path.exists() and trajectories_txt_path.exists() and counterfactuals_path.exists():
        with open(trajectories_pkl_path, "rb") as f:
            trajectories, counterfactuals = pickle.load(f), pd.read_csv(counterfactuals_path)
            print(f"Loaded {len(trajectories)} trajectories from '{trajectories_pkl_path}'.")
            print(f"Loaded {len(counterfactuals)} counterfactuals from '{counterfactuals_path}'.")
            return trajectories, counterfactuals

    # for our analysis, we rely on noise-monotonic simulator
    print(f"Initializing noise-monotonic sepsis simulator.")
    act_noise = UniformNoiseModel(order=Order([i for i in range(SepsisAction.NUM_FULL)]))
    env_noise = SepsisStateUniformNoiseModel(order=Order([i for i in range(State.NUM_TOTAL)]))
    env = Sepsis(dynamics=mdp_path, max_horizon=max_horizon, act_noise_model=act_noise, env_noise_model=env_noise, turn_based=True)

    # function that samples trajectories and enumerates counterfactuals for a given trust level
    def _calculate(trust_level: int, trust_level_id: int, seed: int):
        curr_trajectories, curr_counterfactuals = [], []
        seed_rng = np.random.default_rng(seed)

        # creates AI and clinician agent with target trust level
        ai_agent = AIActor(id=0, policy=ai_policy_path, rng=seed_rng)
        cl_agent = ClinicianActor(id=1, policy=cl_policy_path, trust=trust_level, rng=seed_rng)
        agents = [ai_agent, cl_agent]

        # for trajectory sampling, we rely on the original simulator
        env_simulator = Sepsis(transition_probabilities="./assets/sepsis_transition_probs_original.json", max_horizon=max_horizon, turn_based=True)

        # samples trajectories from the simulator
        curr_trajectories, stats = sample_trajectories(env=env_simulator, agents=agents, num_trajectories=num_trajectories, rng=seed_rng, kind="failure")
        print(f"Sampled {len(curr_trajectories)} failed trajectories for clinician's trust level of {trust_level}.")
        print(f"Total number of sampled trajectories was {stats['success'] + stats['failure']} out of which {stats['success']} had positive outcome.")

        # update identifier of trajectories to not overlap with others
        for t in curr_trajectories: t.id += trust_level_id * num_trajectories

        # remove trajectories for which the approximated distribution is not defined
        with open(mdp_path, "rb") as f: dynamics = pickle.load(f)["transition_matrix"]
        filtered = []
        for t in curr_trajectories:
            include = True
            for time_step in range(t.horizon - 1):
                state = t.states[time_step]
                action = t.actions[time_step][state.player]
                next_state = t.states[time_step + 1]
                if state.player == 1:
                    action = action if action < SepsisAction.NUM_TOTAL else state.act_ai
                    include = include and dynamics[action][state.index][next_state.index] > 0.0
            if include: filtered.append(t)
        curr_trajectories = filtered
        print(f"Skips analysis for {num_trajectories - len(filtered)} trajectories for which the approximated distribution was not defined.")

        # enumerate counterfactuals for each trajectory, considering every alternative action of an active agent for all time-steps
        candidates = []
        for traj in tqdm.tqdm(curr_trajectories, desc="Enumerating possible interventions"):
            # skip trajectories which are too short
            if traj.horizon < 3:
                print(f"Skipping trajectory {traj.id} because of its horizon of {traj.horizon}")
                continue

            # consider all remaining time-steps and active agents
            for time_step in range(0, traj.horizon - 2):
                agent = agents[traj.states[time_step].player]
                act_taken = traj.actions[time_step][agent.id]
                act_alternative = [act for act in env.get_available_actions(traj.states[time_step], agent.id) if act != act_taken]
                for act in act_alternative: candidates.append({"traj": traj, "agent_id": agent.id, "time_step": time_step, "act": act})

        # calculate TCFE for each candidate
        tcfe_seeds = seed_rng.integers(1e5, size=len(candidates))
        tcfe_values = [
            tcfe(trajectory=c["traj"], intervention={c["agent_id"]: c["act"]}, time_step=c["time_step"], env=env, agents=agents, seed=tcfe_seeds[i], num_cf_samples=num_cf_samples)
            for i, c in enumerate(tqdm.tqdm(candidates, desc="Calculating TCFE values"))]

        # persist counterfactuals
        for c, tcfe_value in zip(candidates, tcfe_values):
            curr_counterfactuals.append({"traj_id": c["traj"].id, "agent_id": c["agent_id"], "time_step": c["time_step"], "alternative": c["act"], "trust_level": trust_level, "tcfe": tcfe_value})

        return curr_trajectories, curr_counterfactuals

    # runs calculations in parallel for each wrong total order
    seeds = rng.integers(1e5, size=len(trust_levels))
    results = Parallel(n_jobs=8)(delayed(_calculate)(trust_level, trust_level_id, seed) for trust_level_id, (trust_level, seed) in enumerate(zip(trust_levels, seeds)))
    trajectories = list(itertools.chain(*list(map(lambda r: r[0], results))))
    counterfactuals = list(itertools.chain(*list(map(lambda r: r[1], results))))

    # saves trajectories to disk
    with open(trajectories_pkl_path, "wb") as f:
        pickle.dump(trajectories, f)
        print(f"Saved pickled trajectories under '{trajectories_pkl_path}'.")
    with open(trajectories_txt_path, "w") as f:
        f.write("\n".join([t.render() for t in trajectories]))
        print(f"Saved human-readable trajectories under '{trajectories_txt_path}'.")

    # saves counterfactuals to disk
    df = pd.DataFrame(counterfactuals)
    df.to_csv(counterfactuals_path, index=False)
    print(f"Saved {len(df)} counterfactuals to file '{counterfactuals_path}'.")

    return trajectories, df


def _calculate_causal_quantities(
    artifacts_dir: Path, ai_policy_path: str, cl_policy_path: str, mdp_path: str,
    counterfactuals: pd.DataFrame, trajectories: List[SepsisTrajectory],
    rng: Generator, num_cf_samples: int, max_horizon: int,
) -> pd.DataFrame:
    counterfactuals_w_quantities_path = artifacts_dir / "counterfactuals_w_quantities.csv"

    if counterfactuals_w_quantities_path.exists():
        df = pd.read_csv(counterfactuals_w_quantities_path)
        print(f"Loaded {len(df)} counterfactuals with calculated ASE and PSE quantities from {counterfactuals_w_quantities_path}.")
        return df

    # initialize sepsis noise-monotonic simulator used for analysis
    print(f"Initializing noise-monotonic sepsis simulator.")
    act_noise = UniformNoiseModel(order=Order([i for i in range(SepsisAction.NUM_FULL)]))
    env_noise = SepsisStateUniformNoiseModel(order=Order([i for i in range(State.NUM_TOTAL)]))
    env = Sepsis(dynamics=mdp_path, max_horizon=max_horizon, act_noise_model=act_noise, env_noise_model=env_noise, turn_based=True)

    # function that calculates ASE and PSE for given counterfactuals
    def _calculate(counterfactuals: pd.DataFrame, seed: int):
        df_items = []

        for i, intr in enumerate(tqdm.tqdm(counterfactuals.itertuples(), desc="Calculating ASE and PSE for each selected counterfactual")):
            item = {"traj_id": intr.traj_id, "agent_id": intr.agent_id, "time_step": intr.time_step, "alternative": intr.alternative, "trust_level": intr.trust_level, "tcfe": intr.tcfe}
            trajectory = find_by_id(trajectories, intr.traj_id)

            # creates AI and clinician agent with target trust level
            ai_agent = AIActor(id=0, policy=ai_policy_path)
            cl_agent = ClinicianActor(id=1, policy=cl_policy_path, trust=intr.trust_level)
            agents = [ai_agent, cl_agent]

            # calculates ASE for this intervention
            effect_agents = [1 - intr.agent_id]
            item["ase"] = ase(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative},
                            effect_agents=effect_agents, time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=seed + i)

            # calculate PSE for this intervention
            item["pse"] = pse(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative},
                            time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=seed + i)

            # persists item
            df_items.append(item)

        return df_items

    # runs calculations in parallel, for a chunked dataset
    chunk_size = 500
    chunks = [counterfactuals[i:i + chunk_size] for i in range(0, counterfactuals.shape[0], chunk_size)]
    chunks_seed = rng.integers(1e5, size=len(chunks))
    items = Parallel(n_jobs=mp.cpu_count())(delayed(_calculate)(df, seed) for df, seed in zip(chunks, chunks_seed))
    items = itertools.chain(*items)

    # saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(counterfactuals_w_quantities_path, index=False)
    print(f"Saved {len(df)} counterfactuals with ASE and PSE values to '{counterfactuals_w_quantities_path}'.")
    return df


def _calculate_causal_quantities_with_misspecified_order(
    artifacts_dir: Path, ai_policy_path: str, cl_policy_path: str, mdp_path: str,
    counterfactuals: pd.DataFrame, trajectories: List[SepsisTrajectory],
    rng: Generator, num_cf_samples: int, num_order_shuffles: int, max_horizon: int
) -> pd.DataFrame:
    counterfactuals_w_order_error_path = artifacts_dir / "counterfactuals_w_order_error.csv"

    if counterfactuals_w_order_error_path.exists():
        df = pd.read_csv(counterfactuals_w_order_error_path)
        print(f"Loaded {len(df)} counterfactuals with calculated ASE order error from {counterfactuals_w_order_error_path}.")
        return df

    # a function that randomly samples a wrong env/act total order and calculates ASE/PSE for it
    def _calculate(seed: int, order_label: str):
        seed_rng = np.random.default_rng(seed)
        df_items = []

        # randomly generate a wrong total order
        act_order_items = [i for i in range(SepsisAction.NUM_FULL)]
        seed_rng.shuffle(act_order_items)
        env_order_items = [i for i in range(State.NUM_TOTAL)]
        seed_rng.shuffle(env_order_items)

        # initialize sepsis noise-monotonic simulator used for analysis with wrong total order
        print(f"Initializing noise-monotonic sepsis simulator.")
        act_noise = UniformNoiseModel(order=Order(act_order_items))
        env_noise = SepsisStateUniformNoiseModel(order=Order(env_order_items))
        env = Sepsis(dynamics=mdp_path, max_horizon=max_horizon, act_noise_model=act_noise, env_noise_model=env_noise, turn_based=True)

        # calculates ASE and PSE for each selected counterfactual
        for intr in tqdm.tqdm(counterfactuals.itertuples(), desc="Calculating ASE and PSE for each selected counterfactual with wrong total order assumption"):
            item = {"traj_id": intr.traj_id, "agent_id": intr.agent_id, "time_step": intr.time_step, "alternative": intr.alternative, "order": order_label,
                    "trust_level": intr.trust_level, "tcfe": intr.tcfe, "ase": intr.ase, "pse": intr.pse}
            item_seed = seed_rng.integers(1e5, size=1).item()
            trajectory = find_by_id(trajectories, intr.traj_id)

            # creates AI and clinician agent with target trust level
            ai_agent = AIActor(id=0, policy=ai_policy_path)
            cl_agent = ClinicianActor(id=1, policy=cl_policy_path, trust=intr.trust_level)
            agents = [ai_agent, cl_agent]

            # calculates ASE for this intervention
            effect_agents = [1 - intr.agent_id]
            item["ase_w_error"] = ase(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative},
                                      effect_agents=effect_agents, time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=item_seed)

            # calculate PSE for this intervention
            item["pse_w_error"] = pse(trajectory=trajectory, env=env, agents=agents, intervention={intr.agent_id: intr.alternative},
                                      time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=item_seed)

            # persists item
            df_items.append(item)

        return df_items

    # runs calculations in parallel for each wrong total order
    seeds = rng.integers(1e5, size=num_order_shuffles)
    items = Parallel(n_jobs=8)(delayed(_calculate)(seed, f"order_{i}") for i, seed in enumerate(seeds))
    items = itertools.chain(*items)

    # saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(counterfactuals_w_order_error_path, index=False)
    print(f"Saved {len(df)} counterfactuals with ASE and PSE values to '{counterfactuals_w_order_error_path}'.")
    return df


def _calculate_or_load_posterior_sample_complexity_analysis(
    artifacts_dir: Path, ai_policy_path: str, cl_policy_path: str, mdp_path: str,
    counterfactuals: pd.DataFrame, trajectories: List[SepsisTrajectory],
    rng: Generator, max_horizon: int,
) -> Tuple[pd.DataFrame]:
    df_file = artifacts_dir / "sample_complexity_analysis.csv"

    if df_file.exists():
        df = pd.read_csv(df_file)
        print(f"Loaded ASE/PSE sample complexity analysis from '{df_file}'.")
        return df_file

    candidate_num_samples = np.arange(10, 110, 10).tolist()
    candidate_seeds = rng.integers(1e5, size=10).tolist()

    # initialize sepsis noise-monotonic simulator used for analysis
    print(f"Initializing noise-monotonic sepsis simulator.")
    act_noise = UniformNoiseModel(order=Order([i for i in range(SepsisAction.NUM_FULL)]))
    env_noise = SepsisStateUniformNoiseModel(order=Order([i for i in range(State.NUM_TOTAL)]))
    env = Sepsis(dynamics=mdp_path, max_horizon=max_horizon, act_noise_model=act_noise, env_noise_model=env_noise, turn_based=True)
    agents = {trust_level: [AIActor(id=0, policy=ai_policy_path), ClinicianActor(id=1, policy=cl_policy_path, trust=trust_level)]
              for trust_level in counterfactuals.trust_level.unique()}

    def _calculate(num_cf_samples: int, seed: int):
        df_items = []

        for i, intr in enumerate(tqdm.tqdm(counterfactuals.itertuples())):
            trajectory = find_by_id(trajectories, intr.traj_id)
            ase_curr = ase(trajectory=trajectory, env=env, agents=agents[intr.trust_level], effect_agents=[1 - intr.agent_id],
                            intervention={intr.agent_id: intr.alternative}, time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=seed)
            pse_curr = pse(trajectory=trajectory, env=env, agents=agents[intr.trust_level], intervention={intr.agent_id: intr.alternative},
                            time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=seed)
            tcfe_curr = tcfe(trajectory=trajectory, env=env, agents=agents[intr.trust_level], intervention={intr.agent_id: intr.alternative},
                            time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=seed)
            df_items.append({"intr_id": i, "seed": seed, "num_cf_samples": num_cf_samples, "ase": ase_curr, "pse": pse_curr, "tcfe": tcfe_curr})

        return df_items

    # runs calculations in parallel for each candidate
    items = Parallel(n_jobs=16)(delayed(_calculate)(num_samples, seed) for num_samples, seed in itertools.product(candidate_num_samples, candidate_seeds))
    items = itertools.chain(*items)

    # saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(df_file, index=False)
    print(f"Saved ASE/PSE sample complexity analysis to '{df_file}'.")
    return df


if __name__ == "__main__":
    typer.run(main)
