import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.random import Generator
from typing import Optional, Literal, List, Tuple, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ase.envs.causal import CausalEnv, CausalActor


def sample_trajectories(env: "CausalEnv", agents: List["CausalActor"], num_trajectories: int, kind: Literal["success", "failure", "all"] = "all", rng: Optional[Generator] = None):
    """Samples trajectories from a passed environment.
    Args:
        env (CausalEnv): Environment to sample trajectories from.
        agents (List[&quot;CausalActor&quot;]): List of agents acting in the environment.
        num_trajectories (int): Number of trajectories to sample.
        kind (Literal['success', 'failure', 'all'], optional): Determines the kind of trajectories to sample. Defaults to 'all'.
        rng (Optional[Generator], optional): Fixed random number generator, used for reproducibility. Defaults to None.
    Returns:
        List[Trajectory]: List of sampled trajectories.
    """
    trajectories = []
    stats = {"success": 0, "failure": 0}

    while len(trajectories) < num_trajectories:
        # sample new trajectory
        trajectory = env.sample_trajectory(agents, rng=rng)
        trajectory.id = len(trajectories)

        if kind == "failure" and trajectory.failed():
            trajectories.append(trajectory)
        elif kind == "success" and trajectory.success():
            trajectories.append(trajectory)
        elif kind == "all":
            trajectories.append(trajectory)

        # update stats
        stats["success"] += 1 if trajectory.success() else 0
        stats["failure"] += 1 if trajectory.failed() else 0

    return trajectories, stats


def find_range_for_item(item: float, ranges: List[Tuple[float]]) -> int:
    """Finds a range that contains an item using binary search.
    Args:
        item (float): Item to find.
        ranges (List[Tuple[float]]): List of ranges, sorted in ascending order (e.g., [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0)]).
    Returns:
        int: Index of the range that contains the item
    """
    def _bin_search(low_ind: int, high_ind: int) -> int:
        if low_ind > high_ind:
            return -1

        mid_ind = (high_ind + low_ind) // 2

        if ranges[mid_ind][0] <= item and item < ranges[mid_ind][1]:
            return mid_ind
        elif item < ranges[mid_ind][0]:
            return _bin_search(low_ind, mid_ind - 1)
        else:
            return _bin_search(mid_ind + 1, high_ind)

    return _bin_search(0, len(ranges) - 1)


def get_probability_ranges(probs: List[float]) -> List[Tuple[float]]:
    """Constructs a list of ranges from a probability distribution.
    Args:
        probs (List[float]): Probability distribution (e.g., [0.25, 0.25, 0.50]).
    Returns:
        List[Tuple[float]]: List of probability ranges (e.g., [(0.0, 0.25), (0.25, 0.50), (0.50, 1.0)])
    """
    result, limit = [], 0.0
    for prob in probs:
        result.append((limit, limit + prob))
        limit += prob
    return result


def find_by_id(items: List[Any], id: Any) -> Optional[Any]:
    """Helper function that finds an item in a list by its id.
    Args:
        items (List[Any]): List of items to search.
        id (Any): Item's id to search for.
    Returns:
        Optional[Any]: Item with the given id or None if not found.
    """
    for item in items:
        if hasattr(item, "id") and item.id == id:
            return item
    return None


def export_legend(figure: mpl.figure.Figure, axes: mpl.axes.Axes):
    """Exports a legend from a figure."""
    handles, labels = axes.legend_.legendHandles, [t.get_text() for t in axes.legend_.get_texts()]
    legend = axes.get_legend()
    legend_bbox = legend.get_tightbbox(figure.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(figure.dpi_scale_trans.inverted())
    legend_fig, legend_ax = plt.subplots(figsize=(legend_bbox.width, legend_bbox.height))
    legend_ax.axis("off")
    legend_squared = legend_ax.legend(
        handles, labels,
        bbox_to_anchor=(0, 0, 1, 1),
        bbox_transform=legend_fig.transFigure,
        frameon=True,
        facecolor="#ffffff",
        fancybox=True,
        shadow=False,
        ncol=min(len(labels), 3),
        fontsize=20,
        title=legend.get_title().get_text(),
    )
    return legend_fig, legend_squared
