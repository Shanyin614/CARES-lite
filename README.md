# CARES-lite

**CARES-lite** is a lightweight prototype for adaptive clustered federated learning under synthetic label-skew heterogeneity.

The project studies whether clients with different label preferences can be automatically grouped without using ground-truth group labels during training. CARES-lite first performs a short global FedAvg warm-up, then profiles clients through a small pool of model probes, clusters clients with a DPMM-style Gaussian mixture, and trains one federated model per discovered cluster.

## Highlights

- Automatic client clustering with predicted `K`
- Probe-based client loss profiling
- DPMM-style adaptive cluster discovery
- Clustered FedAvg after warm-up
- Dynamic re-clustering with model inheritance
- Synthetic ground-truth evaluation with ARI, NMI, and Purity
- Fashion-MNIST label-skew benchmark

## Repository structure

```text
CARES-lite/
├── src/
│   ├── main.py          # Entry point
│   ├── config.py        # Command-line arguments
│   ├── data.py          # Client partitioning and synthetic group construction
│   ├── client.py        # FL client logic, local training, probe profiling
│   ├── server.py        # FedAvg, clustering, re-clustering, evaluation
│   └── model.py         # CNN model
├── legacy/              # Historical scripts
├── ifca_results.csv     # IFCA baseline results
├── results_v0.csv       # FedAvg and Loss-DPMM baseline results
├── run.sh               # Shell launcher
├── run.bat              # Windows launcher
└── README.md
展开
Method overview
CARES-lite follows four stages.

1. Synthetic FL data construction
The benchmark uses Fashion-MNIST with 100 clients. Clients are assigned to four latent ground-truth groups:

G0: [0, 2, 6]
G1: [1, 3]
G2: [4, 8]
G3: [5, 7, 9]
Each client mainly receives samples from one group, producing a controlled label-skew federated learning setting.

The ground-truth groups are used only for evaluation metrics such as ARI, NMI, and Purity. They are not used by CARES-lite during training.

2. Global warm-up
A single global model is trained with FedAvg for a few rounds. This gives all clients a shared initialization before clustering.

3. Probe-based client profiling
For each client, CARES-lite evaluates a pool of model probes on the client's validation data. The resulting loss vector is used as a compact profile of the client's data distribution.

The probe pool contains:

1 original model probe
10 class-ablation probes
several random perturbation probes
4. Adaptive clustering and clustered FedAvg
Client profiles are clustered using a DPMM-style mixture model. CARES-lite then trains one group model per discovered cluster.

During training, CARES-lite periodically re-profiles clients and re-runs clustering. When clusters change, group models are inherited by matching new clusters to previous clusters, rather than reinitializing all group models from scratch.

Installation
Create an environment:

conda create -n cares-lite python=3.11
conda activate cares-lite
Install dependencies:

pip install torch torchvision numpy pandas scikit-learn scipy
If you use a newer NVIDIA GPU and see a CUDA compatibility warning, install a PyTorch build that supports your GPU architecture from the official PyTorch installation page.

Recommended run
The current recommended single-seed configuration is:

python -m src.main \
  --seed 42 \
  --num-clients 100 \
  --total-rounds 50 \
  --warmup-rounds 5 \
  --cluster-interval 5 \
  --min-cluster-size 5 \
  --probe-pool-size 16 \
  --probe-sigma 0.05 \
  --clip-norm 5.0 \
  --dpmm-prior 0.03 \
  --dpmm-max-components 10 \
  --client-frac 0.2 \
  --local-epochs 2 \
  --lr 0.01 \
  --output-dir ./output \
  --output-name cares_r50_e2_lr001.csv
On Windows PowerShell:

python -m src.main `
  --seed 42 `
  --num-clients 100 `
  --total-rounds 50 `
  --warmup-rounds 5 `
  --cluster-interval 5 `
  --min-cluster-size 5 `
  --probe-pool-size 16 `
  --probe-sigma 0.05 `
  --clip-norm 5.0 `
  --dpmm-prior 0.03 `
  --dpmm-max-components 10 `
  --client-frac 0.2 `
  --local-epochs 2 `
  --lr 0.01 `
  --output-dir ./output `
  --output-name cares_r50_e2_lr001.csv
Example output
A typical run prints the warm-up phase, the first clustering phase, clustered training rounds, and final metrics:

Phase 0: Initialization
Phase 1: Warm-up Global FedAvg
Transition: First DPMM Clustering
Phase 2: Clustered FedAvg with Dynamic Re-clustering
Evaluation
Final Results
Important metrics:

k_pred: predicted number of clusters
client_avg_acc: average accuracy across clients
micro_acc: global micro accuracy
client_avg_macro_f1: average client-level macro-F1
global_macro_f1: macro-F1 over pooled test predictions
ari: Adjusted Rand Index against synthetic ground-truth groups
nmi: Normalized Mutual Information against synthetic ground-truth groups
purity: cluster purity against synthetic ground-truth groups
Main result
Single-seed result with the recommended configuration:

total_rounds = 50
warmup_rounds = 5
cluster_interval = 5
client_frac = 0.2
local_epochs = 2
lr = 0.01
probe_pool_size = 16
probe_sigma = 0.05
dpmm_prior = 0.03
Method	K setting	K predicted	Micro Acc	Client Avg Acc	Global Macro-F1	Client Avg Macro-F1	ARI	NMI	Purity
CARES-lite	auto	4	0.8847	0.8847	0.8748	0.6742	1.0000	1.0000	1.0000
Additional CARES-lite ablations
Setting	Micro Acc	Global Macro-F1	Client Avg Macro-F1	ARI	NMI	Purity
R30, E=1, LR=0.02, CF=0.2	0.8482	0.8341	0.6273	1.0000	1.0000	1.0000
R50, E=1, LR=0.02, CF=0.2	0.8592	0.8420	0.6228	1.0000	1.0000	1.0000
R50, E=1, LR=0.02, CF=0.3	0.8700	0.8552	0.6617	1.0000	1.0000	1.0000
R50, E=2, LR=0.01, CF=0.2	0.8847	0.8748	0.6742	1.0000	1.0000	1.0000
R50, E=2, LR=0.01, CF=0.3	0.8857	0.8752	0.6974	1.0000	1.0000	1.0000
CF denotes client fraction. E denotes local epochs.

The CF=0.3 setting slightly improves global performance and noticeably improves client-level macro-F1, but it requires 50% more participating clients per round. Therefore, the recommended main configuration uses client_frac=0.2.

Baseline results
Existing baseline results in this repository:

Method	K setting	K predicted	Micro Acc	Client Avg Acc	Global Macro-F1	Client Avg Macro-F1
FedAvg	-	-	0.7166	0.7166	0.6386	0.5344
Loss-DPMM + ClusterFedAvg	auto	4	0.8272	0.8272	0.7891	0.5682
IFCA	2	2	0.7946	0.7946	0.7662	0.6143
IFCA	3	3	0.8184	0.8184	0.7798	0.5986
IFCA-oracleK	4	4	0.8275	0.8275	0.8062	0.5500
IFCA	5	4	0.8287	0.8287	0.8103	0.5843
IFCA	6	4	0.8276	0.8276	0.8106	0.5720
IFCA	8	4	0.8019	0.8019	0.7778	0.5588
Compared with the strongest IFCA runs, CARES-lite achieves higher accuracy and macro-F1 while automatically recovering the true number of latent groups.

Notes on interpretation
This project is a lightweight research prototype. The current benchmark is a controlled synthetic label-skew setting on Fashion-MNIST.

Important cautions:

The reported CARES-lite result is currently single-seed.
ARI, NMI, and Purity use synthetic ground-truth group labels for evaluation only.
Ground-truth group labels are not used by CARES-lite during training.
Intermediate evaluation logs are useful for debugging and analysis, but final test results should be reported separately from validation-based tuning.
The current experiment mainly demonstrates recovery of label-skew groups, not arbitrary real-world client heterogeneity.
Reproducibility
The main run saves a CSV file under output/.

Example:

output/cares_r50_e2_lr001.csv
The output contains final metrics such as:

k_pred
client_avg_acc
micro_acc
client_avg_macro_f1
global_macro_f1
ari
nmi
purity
For future multi-seed evaluation, run the same configuration with different seeds and report mean ± standard deviation.

Suggested seeds:

1, 2, 3, 4, 5
Citation
This repository is currently a research prototype. If you use or extend it, please cite the repository or the associated report once available.

License
No license has been specified yet.