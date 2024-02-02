# Agent-Specific Effects

This repository contains the code and instructions needed to replicate the results from the paper "Agent-Specific Effects: A Causal Effect Propagation Analysis in Multi-Agent MDPs".

## Reproducibility

To reproduce the results of the paper, you will need to do the following:

- **Generate the multi-agent MDP's parameters for the sepsis environment**. To do so, run the `learn_sepsis_mdp.ipynb` notebook, which will save the results under `results/sepsis/mdp_ai.pkl` and `results/sepsis/mdp_original.pkl`.
- **Learn the AI and clinician policies**. To do so, run the `learn_sepsis_actors.ipynb` notebook, which will train and save the policies under `results/sepsis/ai_policy.pkl` and `results/sepsis/cl_policy.pkl`.
- To reproduce the **results** for the *graph* environment, it suffices to run the following command:
  ```bash
  python -m ase.scripts.graph_experiment 8854 \
      --artifacts-dir results/graph/ \
      --tcfe-threshold 0.75 \
      --num-trajectories 500 \
      --num-agents 6 \
      --num-cf-samples 100 \
      --num-effect-agents-choices 31 \
      --posterior-sample-complexity 500
  ```
- To **visualize** the results for the graph environment, run the `graph_results.ipynb` notebook.
- To reproduce the **results** for the *sepsis* environment, it suffices to run the following command:
  ```bash
  python -m ase.scripts.sepsis_experiment 8854 \
      --artifacts-dir results/sepsis \
      --mdp-path results/sepsis/mdp_original.pkl \
      --cl-policy-path results/sepsis/cl_policy.pkl \
      --ai-policy-path results/sepsis/ai_policy.pkl \
      --tcfe-threshold 0.8 \
      --num-trajectories 100 \
      --num-cf-samples 100 \
      --max-horizon 40 \
      --shuffle-total-order 5 \
      --trust-values 0.0,0.2,0.4,0.6,0.8,1.0 \
      --posterior-sample-complexity 500
  ```
- To **visualize** the results for the sepsis environment, run the `sepsis_results.ipynb` notebook.

The **time** needed to learn the MMDP's parameters is ~5 hours, whereas the time needed to run the experiment for graph and sepsis is ~2.5 and ~7.5 hours, respectively.

## Dependencies

This project is dependent on Python version 3.9.13. To install the necessary dependencies, it suffices to run `pip install -r requirements.txt`.
