## Chromatography

This project aims to develop RL algorithms for chromatography, specifically for the separation of mixtures into their components.

> It is an experimental project, and probably the first attempt in applying RL to chromatographic separations.

The current implementation solves a relatively simple problem (see below), in a simualted chromatographic environment. There is a lot more to consider, and much room for improvements. 

So far, a PPO agent has been trained to improve separations of random mixtures of 10 to 20 compounds, based on chromatograms as observations. The chromatograms consist of peaks (corresponding to compounds) where the peak apices correspond to retention times. Peak widths and ampitudes are randomly generated, and minor noise has been added to the chromatograms.

### How to run

1. Open the command line in the root directory, and make sure you are your desired Python environment.
2. Install the **reinforceable** package: `pip install -e .`
3. Install **jupyter**, **matplotlib** and **tqdm**: `pip install jupyter matplotlib tqdm`
4. Navigate to `applications/chromatography/` and run jupyter: `jupyter notebook`.
5. Open **run.ipynb** and run all the cells in it.
6. To observe training progression, go back to the command line and run tensorboard: `tensorboard --logdir ./logs/`.

### Files

- **env.py**, the chromatographic environment.
- **env_utils.py**, any utilities needed for the chromatographic environment.
- **networks.py**, the encoder, policy and value networks of a PPO agent.

The agent used is an agent from the **reinforceable** package.

