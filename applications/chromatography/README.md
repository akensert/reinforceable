## Chromatography

This project aims to develop RL algorithms for chromatography, specifically for the separation of mixtures into their components.

> This is an experimental project, and probably the first attempt in applying RL for chromatographic separations.

The current implementation solves a relatively simple problem (see below), in a simualted chromatographic environment. There is a lot more to consider, and much room for improvement. 

So far, a PPO agent has been trained to improve separations of random mixtures of 10 to 20 compounds, based on chromatograms as observations. The chromatograms consist of peaks (corresponding to compounds) where the peak apices correspond to retention times. Peak widths and ampitudes are randomly generated, and minor noise has been added to the chromatograms.

### How to run

1. Navigate to a desired location (via command line), and clone the repository: `git clone git@github.com:akensert/reinforceable.git`
2. Make sure you are in your desired Python environment, and then install the **reinforceable** package: `pip install -e .`
3. Install **jupyter** and **matplotlib**: `pip install jupyter matplotlib`
4. Navigate to `applications/chromatography/` and run jupyter: `jupyter notebook`.
5. Open **run.ipynb** and run all the cells in it.
6. To observe training progression, go back to the command line (you should be in `applications/chromatography/`) and run tensorboard: `tensorboard --logdir ./logs`.

### Files

- **env.py**, the chromatographic environment.
- **env_utils.py**, any utilities needed for the chromatographic environment.
- **networks.py**, the encoder, policy and value networks of the PPO agent.

The agent used is an agent from the **reinforceable** package.

