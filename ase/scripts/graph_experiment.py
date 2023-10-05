import time
import tqdm
import typer
import pickle
import itertools
import numpy as np
import pandas as pd
import multiprocessing as mp

from pathlib import Path
from typing import List, Tuple
from rich import print
from numpy.random import Generator
from joblib import Parallel, delayed
from typing_extensions import Annotated

from ase.envs.graph import Graph, GraphTrajectory, BalancedLevelReward
from ase.actors.graph import GraphActor, GraphAction
from ase.tools.noise import UniformNoiseModel
from ase.tools.order import Order
from ase.tools.algorithms import tcfe, ase
from ase.tools.utils import sample_trajectories, find_by_id


def main(
    seeds: Annotated[List[int], typer.Argument(help="List of random seeds to use for repeated experiment runs.")],
    artifacts_dir: Annotated[Path, typer.Option(help="Path to directory where artifacts are stored. There will be one subdirectory per random seed.", dir_okay=True, file_okay=False)],
    tcfe_threshold: Annotated[float, typer.Option(help="Minimum TCFE an intervention needs to have to be considered for analysis.")] = 0.8,
    posterior_sample_complexity: Annotated[int, typer.Option(help="Run the posterior sample complexity analysis for given number of counterfactuals.")] = 500,
    num_trajectories: Annotated[int, typer.Option(help="Number of trajectories to sample for analysis.")] = 100,
    num_agents: Annotated[int, typer.Option(help="Number of graph environment agents.")] = 6,
    num_cf_samples: Annotated[int, typer.Option(help="Number of counterfactual samples to draw when calculating ASE/TCFE.")] = 100,
    num_effect_agents_choices: Annotated[int, typer.Option(help="Number of subsets of effects agents to consider when running prob/order error analysis.")] = 10,
):
    time_start = time.time()

    for seed in seeds:
        # sets up training artifacts and seed
        artifacts = artifacts_dir / str(seed)
        artifacts.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        print(f"Running experiment for seed {seed}.")

        # creates environment and agents
        order = Order([GraphAction.up.value, GraphAction.down.value, GraphAction.straight.value])
        agents = [GraphActor(id=id) for id in range(num_agents)]
        env = Graph(
            num_agents=num_agents,
            num_levels=3, num_columns=3,
            act_noise_model=UniformNoiseModel(order=order),
            reward_constraint=BalancedLevelReward())

        # generates trajectories, if they're not found in the artifacts directory
        trajectories = _sample_or_load_trajectories(artifacts_dir=artifacts, num_trajectories=num_trajectories, env=env, agents=agents, rng=rng)

        # enumerate all possible counterfactuals and calculate their TCFE
        counterfactuals = _enumerate_or_load_counterfactuals(artifacts_dir=artifacts, trajectories=trajectories, env=env, agents=agents,num_cf_samples=num_cf_samples, rng=rng)

        # runs posterior sample complexity analysis, if requested
        if posterior_sample_complexity:
            _ = _calculate_or_load_posterior_sample_complexity_analysis(artifacts_dir=artifacts, counterfactuals=counterfactuals.sample(posterior_sample_complexity, random_state=rng),
                                                                        trajectories=trajectories, env=env, agents=agents, rng=rng)

        # select only those interventions with satisfying specified TCFE threshold
        counterfactuals = counterfactuals.loc[np.where((counterfactuals.tcfe >= tcfe_threshold))]
        print(f"Selected {len(counterfactuals)} counterfactuals with TCFE >= {tcfe_threshold} for analysis.")

        # calculate ASE for each intervention and each possible combination of effect-agents
        counterfactuals_w_ase = _calculate_or_load_ground_truth_ase(artifacts_dir=artifacts, counterfactuals=counterfactuals, trajectories=trajectories,
                                                                    env=env, agents=agents, num_cf_samples=num_cf_samples, rng=rng)

        # calculate ASE with respect to the agent's probability error
        prob_errors = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
        _ = _calculate_or_load_ase_w_prob_error(artifacts_dir=artifacts, counterfactuals=counterfactuals_w_ase, trajectories=trajectories, env=env, prob_errors=prob_errors,
                                                agents=agents, rng=rng, num_effect_agents_choices=num_effect_agents_choices, num_cf_samples=num_cf_samples)

        # calculate ASE with respect to wrong total order
        wrong_orders = [
            Order([GraphAction.up.value, GraphAction.straight.value, GraphAction.down.value]),
            Order([GraphAction.down.value, GraphAction.up.value, GraphAction.straight.value]),
            Order([GraphAction.down.value, GraphAction.straight.value, GraphAction.up.value]),
            Order([GraphAction.straight.value, GraphAction.up.value, GraphAction.down.value]),
            Order([GraphAction.straight.value, GraphAction.down.value, GraphAction.up.value]),
        ]
        _ = _calculate_or_load_ase_w_order_error(artifacts_dir=artifacts, counterfactuals=counterfactuals_w_ase, trajectories=trajectories,
                                                 orders=wrong_orders, agents=agents, num_effect_agents_choices=num_effect_agents_choices,
                                                 rng=rng, num_cf_samples=num_cf_samples)

    print(f"Experiment finished. Time elapsed: {time.time() - time_start} seconds.")


def _sample_or_load_trajectories(artifacts_dir: Path, num_trajectories: int, env: Graph, agents: List[GraphActor], rng: Generator) -> List[GraphTrajectory]:
    trajectories_pkl_path = artifacts_dir / "trajectories.pkl"
    trajectories_txt_path = artifacts_dir / "trajectories.txt"

    if trajectories_pkl_path.exists() and trajectories_txt_path.exists():
        with open(trajectories_pkl_path, "rb") as f:
            trajectories = pickle.load(f)
            print(f"Loaded {len(trajectories)} trajectories from '{trajectories_pkl_path}'.")
            return trajectories

    trajectories, stats = sample_trajectories(env=env, agents=agents, num_trajectories=num_trajectories, kind="failure", rng=rng)
    print(f"Sampled {len(trajectories)} failed trajectories.")
    print(f"Total number of sampled trajectories was {stats['success'] + stats['failure']} out of which {stats['success']} had positive outcome.")

    with open(trajectories_pkl_path, "wb") as f:
        pickle.dump(trajectories, f)
        print(f"Saved pickled trajectories under '{trajectories_pkl_path}'.")
    with open(trajectories_txt_path, "w") as f:
        f.write("\n".join([t.render() for t in trajectories]))
        print(f"Saved human-readable trajectories under '{trajectories_txt_path}'.")

    return trajectories


def _enumerate_or_load_counterfactuals(artifacts_dir: Path, env: Graph, agents: List[GraphActor], trajectories: List[GraphTrajectory], rng: Generator, num_cf_samples: int = 100) -> pd.DataFrame:
    df_file = artifacts_dir / "counterfactuals.csv"
    df_type = {"traj_id": int, "agent_id": int, "time_step": int, "alternative": int, "tcfe": float}

    if df_file.exists():
        df = pd.read_csv(df_file).astype(df_type)
        print(f"Loaded {len(df)} counterfactuals from file '{df_file}'.")
        return df

    # enumerates all counterfactuals of the given trajectory and calculates their TCFE
    def _calculate(trajectory: GraphTrajectory, seed: int):
        df_items = []
        seed_rng = np.random.default_rng(seed)

        for time_step in range(0, trajectory.horizon - 2):
            for agent in agents:
                act_taken = trajectory.actions[time_step][agent.id]
                act_alternative = [act for act in env.get_available_actions(trajectory.states[time_step], agent.id, strict=True) if act != act_taken]
                act_seeds = seed_rng.integers(1e5, size=len(act_alternative)).tolist()
                act_tcfe = Parallel(n_jobs=1)(delayed(tcfe)(
                    trajectory=trajectory, env=env, agents=agents, intervention={agent.id: act}, time_step=time_step,
                    num_cf_samples=num_cf_samples, seed=seed) for act, seed in zip(act_alternative, act_seeds))
                for act, act_tcfe_value in zip(act_alternative, act_tcfe):
                    df_items.append({"traj_id": trajectory.id, "agent_id": agent.id, "time_step": time_step, "alternative": act, "tcfe": act_tcfe_value})

        print(f"Processed trajectory with id {trajectory.id}")
        return df_items

    # Runs calculations in parallel for each trajectory
    seeds = rng.integers(1e5, size=len(trajectories)).tolist()
    items = Parallel(n_jobs=mp.cpu_count())(delayed(_calculate)(traj, seed) for traj, seed in zip(trajectories, seeds))
    items = itertools.chain(*items)

    # Saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(df_file, index=False)
    print(f"Saved {len(df)} counterfactuals to file '{df_file}'.")

    return df


def _calculate_or_load_ground_truth_ase(artifacts_dir: Path, counterfactuals: pd.DataFrame, trajectories: List[GraphTrajectory], env: Graph, agents: List[GraphActor], rng: Generator, num_cf_samples: int = 100) -> pd.DataFrame:
    df_file = artifacts_dir / "counterfactuals_w_ase.csv"

    if df_file.exists():
        print(f"Loaded counterfactuals with calculated ASE from '{df_file}'.")
        return pd.read_csv(df_file)

    # calculates ASE for wanted subset of effect-agents
    def _calculate(counterfactuals: pd.DataFrame, seed: int):
        df_items = []
        seed_rng = np.random.default_rng(seed)

        for intr in tqdm.tqdm(counterfactuals.itertuples(), desc="Calculating ground-truth ASE for each selected counterfactual"):
            item = {"traj_id": intr.traj_id, "agent_id": intr.agent_id, "time_step": intr.time_step, "alternative": intr.alternative, "tcfe": intr.tcfe}
            effect_agents = [agent.id for agent in agents if agent.id != intr.agent_id]

            # enumerate all possible combinations of effect-agents, namely
            # (5 choose 1) + (5 choose 2) + (5 choose 3) + (5 choose 4) + (5 choose 5)
            # amounting to 31 possible combinations
            effect_agents_choices = list(itertools.chain(*[itertools.combinations(effect_agents, k) for k in range(1, len(effect_agents) + 1)]))

            ea_seeds = seed_rng.integers(1e5, size=len(effect_agents_choices)).tolist()
            ea_labels = [f"ase_{','.join([f'ag{agent_id}' for agent_id in c])}" if len(c) != len(effect_agents) else "ase_total" for c in effect_agents_choices]
            ea_values = Parallel(n_jobs=1)(delayed(ase)(
                    trajectory=trajectories[intr.traj_id], env=env, agents=agents, effect_agents=c, intervention={intr.agent_id: intr.alternative},
                    time_step=intr.time_step, seed=seed, num_cf_samples=num_cf_samples) for c, seed in zip(effect_agents_choices, ea_seeds))
            for label, value in zip(ea_labels, ea_values): item[label] = value

            df_items.append(item)

        return df_items

    # runs calculations in parallel, for a chunked dataset
    chunk_size = 50
    chunks = [counterfactuals[i:i + chunk_size] for i in range(0, counterfactuals.shape[0], chunk_size)]
    chunks_seed = rng.integers(1e5, size=len(chunks))
    items = Parallel(n_jobs=mp.cpu_count())(delayed(_calculate)(df, seed) for df, seed in zip(chunks, chunks_seed))
    items = itertools.chain(*items)

    # Saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(df_file, index=False)
    print(f"Saved counterfactuals with calculated ASE to '{df_file}'.")

    return df


def _calculate_or_load_posterior_sample_complexity_analysis(artifacts_dir: Path, counterfactuals: pd.DataFrame, trajectories: List[GraphTrajectory], env: Graph, agents: List[GraphActor], rng: Generator) -> Tuple[pd.DataFrame]:
    df_file = artifacts_dir / "sample_complexity_analysis.csv"

    if df_file.exists():
        df = pd.read_csv(df_file)
        print(f"Loaded TCFE/ASE sample complexity analysis from '{df_file}'.")
        return df_file

    candidate_num_samples = np.arange(10, 110, 10).tolist()
    candidate_seeds = rng.integers(1e5, size=10).tolist()

    # calculate ASE/TCFE for a given number of posterior samples
    def _calculate(num_cf_samples: int, seed: int):
        df_items = []

        for i, intr in enumerate(tqdm.tqdm(counterfactuals.itertuples(), desc=f"Calculating TCFE/ASE sample complexity for candidate {num_cf_samples}")):
            trajectory = find_by_id(trajectories, intr.traj_id)
            ase_curr = ase(trajectory=trajectory, env=env, agents=agents, effect_agents=[agent.id for agent in agents if agent.id != intr.agent_id],
                           intervention={intr.agent_id: intr.alternative}, time_step=intr.time_step, num_cf_samples=num_cf_samples, seed=seed)
            tcfe_curr = tcfe(trajectory=trajectories[intr.traj_id], env=env, agents=agents, intervention={intr.agent_id: intr.alternative}, time_step=intr.time_step,
                             num_cf_samples=num_cf_samples, seed=seed)
            df_items.append({"intr_id": i, "seed": seed, "num_cf_samples": num_cf_samples, "tcfe": tcfe_curr, "ase": ase_curr})

        return df_items

    # Runs calculations in parallel for each candidate
    items = Parallel(n_jobs=mp.cpu_count())(delayed(_calculate)(num_samples, seed) for num_samples, seed in itertools.product(candidate_num_samples, candidate_seeds))
    items = itertools.chain(*items)

    # Saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(df_file, index=False)
    print(f"Saved TCFE/ASE sample complexity analysis to '{df_file}'.")

    return df


def _calculate_or_load_ase_w_prob_error(artifacts_dir: Path, counterfactuals: pd.DataFrame, trajectories: List[GraphTrajectory], env: Graph, agents: List[GraphActor], prob_errors: List[float], rng: Generator, num_effect_agents_choices: int = 10, num_cf_samples: int = 100) -> pd.DataFrame:
    df = counterfactuals.copy()
    df_file = artifacts_dir / "counterfactuals_w_ase_prob_error.csv"

    if df_file.exists():
        print(f"Loaded counterfactuals with calculated ASE prob. error from '{df_file}'.")
        return pd.read_csv(df_file)

    # calculate ASE for a given probability error and specified subset of effect-agents
    def _calculate(err_max: float, seed: int):
        df_items = []
        seed_rng = np.random.default_rng(seed)

        for i, item in tqdm.tqdm(enumerate(counterfactuals.to_dict(orient="records")), desc=f"Calculating ASE with prob. error of {err_max}"):
            effect_agents = [agent.id for agent in agents if agent.id != item["agent_id"]]
            effect_agents_choices = list(itertools.chain(*[itertools.combinations(effect_agents, k) for k in range(1, len(effect_agents) + 1)]))
            effect_agents_choices = [effect_agents_choices[i] for i in seed_rng.choice(len(effect_agents_choices), size=num_effect_agents_choices, replace=False)]

            # sample agent's p_i with error from range [p_i - err_max, p_i + err_max]
            errors = [seed_rng.uniform(max(0, agent.p_i - err_max), min(agent.p_i + err_max, 1)) for agent in agents]
            agents_with_error = [GraphActor(id=id, p_i=error) for id, error in enumerate(errors)]

            # calculate agent-specific effect for agents with error
            ase_seeds = seed_rng.integers(1e5, size=len(effect_agents_choices)).tolist()
            ase_labels = [f"ase_{','.join([f'ag{agent_id}' for agent_id in c])}" if len(c) != len(effect_agents) else "ase_total" for c in effect_agents_choices]
            ase_ground_truth = [item[label] for label in ase_labels]
            ase_w_error = [ase(
                trajectory=trajectories[item["traj_id"]], env=env, agents=agents_with_error, effect_agents=c,
                intervention={item["agent_id"]: item["alternative"]}, time_step=item["time_step"], seed=seed, num_cf_samples=num_cf_samples)
                for c, seed in zip(effect_agents_choices, ase_seeds)]

            # append result to calculate statistics; note that we do not average the ase error over the
            # selected effect agents here, but defer that to the evaluation notebook
            for label, ase_true, ase_error in zip(ase_labels, ase_ground_truth, ase_w_error):
                df_items.append({
                    "traj_id": item["traj_id"], "agent_id": item["agent_id"], "int_id": i, "err_max": err_max,
                    "ase_type": label, "ase_true": ase_true, "ase_w_error": ase_error, "tcfe": item["tcfe"],
                    "time_step": item["time_step"], "seed": item["seed"], "alternative": item["alternative"]})

        return df_items

    # runs calculations in parallel for each candidate
    seeds = rng.integers(1e5, size=len(prob_errors)).tolist()
    items = Parallel(n_jobs=mp.cpu_count())(delayed(_calculate)(err_max, seed) for err_max, seed in zip(prob_errors, seeds))
    items = itertools.chain(*items)

    # saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(df_file, index=False)
    print(f"Saved counterfactuals with calculated ASE prob. error to '{df_file}'.")

    return df


def _calculate_or_load_ase_w_order_error(artifacts_dir: Path, counterfactuals: pd.DataFrame, trajectories: List[GraphTrajectory], agents: List[GraphActor], orders: List[Order], rng: Generator, num_effect_agents_choices: int = 10, num_cf_samples: int = 100) -> pd.DataFrame:
    df = counterfactuals.copy()
    df_file = artifacts_dir / "counterfactuals_w_ase_order_error.csv"

    if df_file.exists():
        print(f"Loaded counterfactuals with calculated ASE order error from '{df_file}'.")
        return pd.read_csv(df_file)

    # calculates ASE for a given total order and specified subset of effect-agents
    def _calculate(order: Order, seed: int):
        df_items = []
        rng_seed = np.random.default_rng(seed)

        for i, item in tqdm.tqdm(enumerate(counterfactuals.to_dict(orient="records")), desc=f"Calculating ASE with order {str(order)}"):
            env_w_wrong_order = Graph(
                num_agents=len(agents),
                num_levels=3, num_columns=3,
                act_noise_model=UniformNoiseModel(order=order),
                reward_constraint=BalancedLevelReward())

            effect_agents = [agent.id for agent in agents if agent.id != item["agent_id"]]
            effect_agents_choices = list(itertools.chain(*[itertools.combinations(effect_agents, k) for k in range(1, len(effect_agents) + 1)]))
            effect_agents_choices = [effect_agents_choices[i] for i in rng_seed.choice(len(effect_agents_choices), size=num_effect_agents_choices, replace=False)]

            # calculate agent-specific effect with wrong total order
            ase_seeds = rng_seed.integers(1e5, size=len(effect_agents_choices)).tolist()
            ase_labels = [f"ase_{','.join([f'ag{agent_id}' for agent_id in c])}" if len(c) != len(effect_agents) else "ase_total" for c in effect_agents_choices]
            ase_ground_truth = [item[label] for label in ase_labels]
            ase_w_error = Parallel(n_jobs=1)(delayed(ase)(
                trajectory=trajectories[item["traj_id"]], env=env_w_wrong_order, agents=agents, effect_agents=c,
                intervention={item["agent_id"]: item["alternative"]}, time_step=item["time_step"], seed=seed, num_cf_samples=num_cf_samples)
                for c, seed in zip(effect_agents_choices, ase_seeds))

            # append result to calculate statistics; note that we do not average the ase error over the
            # selected effect agents here, but defer that to the evaluation notebook
            for label, ase_true, ase_error in zip(ase_labels, ase_ground_truth, ase_w_error):
                df_items.append({
                    "traj_id": item["traj_id"], "agent_id": item["agent_id"], "int_id": i, "order": str(order),
                    "ase_type": label, "ase_true": ase_true, "ase_w_error": ase_error, "tcfe": item["tcfe"],
                    "time_step": item["time_step"], "seed": item["seed"], "alternative": item["alternative"]})

        return df_items

    # runs calculations in parallel for each wrong total order
    seeds = rng.integers(1e5, size=len(orders)).tolist()
    items = Parallel(n_jobs=mp.cpu_count())(delayed(_calculate)(order, seed) for order, seed in zip(orders, seeds))
    items = itertools.chain(*items)

    # saves calculated values to disk
    df = pd.DataFrame(items)
    df.to_csv(df_file, index=False)
    print(f"Saved counterfactuals with calculated ASE order error to '{df_file}'.")
    return df


if __name__ == "__main__":
    typer.run(main)
