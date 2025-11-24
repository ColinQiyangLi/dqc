# Experiment Data

We release the experiment data `dqc-exp-data.pkl` to facilitate future research. Each file is a dictionary keyed by `(task_name, method_name)` tuples. Each entry contains an numpy array of shape `(4, 10)`. The array stores the success rate (averaged over 5 tasks) at 250K, 500K, 750K, 1M training steps for 10 seeds.

See [sanity-check.ipynb](sanity-check.ipynb) for a quick example for retrieving the success rate at 1M training steps.
