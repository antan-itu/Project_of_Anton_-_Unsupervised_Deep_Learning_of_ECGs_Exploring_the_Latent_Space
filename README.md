# Unsupervised Deep Learning of ECGs: Exploring the Latent Space

**Author:** Anton

**Supervisors:** Veronika & Jørgen 

**Context:** Master's Thesis Project

## Project Overview
This repository contains the codebase for my Master's Thesis, investigating the use of CNN-based autoencoders to explore the latent space of electrocardiograms (ECGs). The models are trained and evaluated using the MIMIC-IV dataset, aiming to uncover underlying structural and rhythmic patterns (such as Atrial Fibrillation) through unsupervised learning.

## Repository Structure

The project is organized into the following main directories:

* **`/exploration_and_preparation/`**
  * `src/`: Contains scripts for the initial exploration, preprocessing, and splitting of the MIMIC-IV dataset.

* **`/model_development/`**
  * `notebooks/`: Jupyter notebooks from the initial development phases and prototyping.
  * `src/`: Core Python scripts representing different iterations of the autoencoder architectures, training loops, and classification heads.
  * *Scripts:* For exploring the latent space (UMAP generation) and visualizing hyperparameter performance.
  * *Results:* Exported metrics, and logs from the various model runs.

* **`/weekly_meeting_material/`**
  * A storage directory for plots, diagrams, and other visual materials the meetings.

* **`weekly_meetings.md`**
  * A log summarizing weekly achievements, current struggles, and specific discussion points for my meetings with Veronika and Jørgen.