# Unsupervised Deep Learning of ECGs: Exploring the Latent Space

**Author:** Anton

**Supervisors:** Veronika & Jørgen 

**Context:** Master's Thesis Project

## Project Overview
This repository contains the codebase for my Master Thesis, investigating the use of CNN-based autoencoders to explore the latent space of electrocardiograms (ECGs). The models are trained and evaluated on the MIMIC-IV dataset, aiming to uncover underlying structural and rhythmic patterns atrial fibrillation through unsupervised learning.

**Abstract -** Cardiologists rely on electrocardiograms (ECGs) to diagnose arrhythmias such as atrial fibrillation (AF), yet manual interpretation and rule-based algorithms do not scale to large workloads and remain prone to errors. Recent studies in deep learning have shown strong performance in this domain. To further investigate the potential of unsupervised machine learning, this study explores whether a convolutional neural network based autoencoder can extract clinically meaningful features by exploring the latent space and using these representations for AF detection. 

We trained the architecture on the MIMIC-IV dataset that contains approximately 800,000 12-lead 10-second ECGs. To assess the clinical utility of the bottleneck, we evaluated both the reconstruction error and classification performance, using logistic regression and XGBoost. 

The model with the best classification performance achieved a ROC-AUC of 0.955 and a PR-AUC of 0.488, exceeding the AF prevalence of 1.8\%. Visual inspection of the latent space revealed a manifold where AF cases were scattered along the outside, rather than forming distinct clusters. Furthermore, the model showed a reconstruction RMSE of 0.289 and a $R^2$ of 0.916, while adjusting the hyperparameters demonstrated a trade-off between reconstruction and linear separability.

Our findings show that autoencoders can extract clinical features, but the current classification performance remains insufficient as a clinical screening tool. The unsupervised explorative approach offers a transparent and scalable way to compress the most important features into the latent space, while simultaneously functioning as a denoiser. Future work should focus on improving the label quality, optimizing the model for raw ECG signals, and evaluating generalizability across a broader range of cardiovascular conditions.

## Repository Structure

The project is organized into the following main directories:

* **`/exploration_and_preparation/`**
  * `src/`: Contains scripts for the initial exploration, preprocessing, and splitting of the MIMIC-IV dataset. The folders also contains scripts for a manual review of 400 ECGs

* **`/model_development/`**
  * `notebooks/`: Jupyter notebooks from the initial development phases and prototyping.
  * `src/`: Core Python scripts representing different iterations of the autoencoder architectures, training loops, and classification heads.
  * `results/`: Contains mainly metrics and logs from various model runs. The folder also contains results from the manual review, UMAP visualisation scripts and other result scripts for the report.

* **`/weekly_meeting_material/`**
  * A storage directory for plots, diagrams, and other visual materials for the weekly meetings.

* **`weekly_meetings.md`**
  * A log summarizing weekly achievements, current struggles, and specific discussion points for my meetings with Veronika and Jørgen.

* **`src/.old/`**
  * These folders are used for previous iterations of the project, for testing, prototyping and experimenting with several different approaches.