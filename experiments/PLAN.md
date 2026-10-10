# Experiment matrix

1. Centralized logistic regression and centralized MLP (reference baselines)
2. Independent bank-local MLPs
3. FedAvg on IID vs non-IID partitions; optionally FedProx
4. Networked sync FL vs round-manager with quorum/deadline rules
5. Pod kill, slow bank, delayed updates, insufficient quorum
6. 3 versus 5 clients (later extend partition function)

Track training run seed, dataset version, partition parameters, class prevalence, hardware, number of rounds, accuracy-independent fraud metrics, mean/p95 round time, model bytes transferred, restart recovery duration, completed rounds, CPU and memory. Keep holdout chronological and untouched.
