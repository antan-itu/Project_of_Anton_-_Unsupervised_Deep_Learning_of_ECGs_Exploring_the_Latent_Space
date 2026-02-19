# Anton's Weekly Meetings

* [19 February 2026](#date-19-february-2026)
* [05 February 2026](#date-05-february-2026)
* [Template](#date-template)

---
### Date: 19 February 2026

#### Who did you help this week?

* N/A 

#### What helped you this week?

* Being at the KU office and getting help from Jonas and Jørgen to set up access to the server for training the models.
* Daily discussions about the results with Jørgen.
* Feedback from this week's supervision, which provided a lot of guidance for future work.

#### What did you achieve?

Previous weeks todo list:
* Called SAP and registered supervisors.
* Set up the GitHub repository and Overleaf Project with matching titles.
* Wrote the Problem Statement and got it approved.
* Reviewed the guidelines for writing in preparation for the Overleaf document.
* Checked the consistency of the ECG files and created a log-file.

Other achievements:
* Built a basic CNN-autoencoder and tested it on 5,000 and 800,000 ECGs.
* Completed 16 runs with 5,000 ECGs - the model prefers higher dimensionality and larger filters.

#### What did you struggle with?

* The model crashed when training on the full dataset - this is hopefully solved by lowering the batch size to 128.
* The loss curve looks "too good," Veronica suggests a side experiment to determine how few cases are too few.
* Needed clarification on how to structure flowcharts for methods and inclusion/exclusion criteria.

#### What would you like to work on next week?

* Continue the Literature Review using tools like Lens, PubMed, and OpenAlex.
* Implement cross-validation - An 80/20 split on 5,000 ECGs can give misleadingly results, since the model can recieve a "easy" or noisy batch. Cross‑validation averages across many splits, making it possible to check if the models are significantly different.
* Experiment with different hyperparameters using ([RandomizedSearchCV](https://medium.com/@bhagyarana80/tuning-hyperparameters-like-a-pro-with-gridsearchcv-and-randomizedsearchcv-611565c0e551)), testing higher dimensionality, kernels, filters, pooling, dropout rates, and strides.
* Add labels to the UMAP projection by updating the metadata without rerunning the UMAP.
* Establish the hold out set of 150,000 ECGs using stratification to maintain similar distributions across age groups and gender.
* Draft flowcharts for the methods and inclusion/exclusion criteria, referencing the ([PRISMA-Statement](https://www.prisma-statement.org/)).

#### Where do you need help from Veronika?

* Feedback on my work once I've implemented cross-validation, RandomizedSearchCV, and added labels to the UMAP.
* Discussing and getting feedback on the flowcharts for the methods and the inclusion/exclusion criteria.
* Discussing existing papers to include in the Literature Review.
* Guidance on stratifying the 150,000 ECG hold-out dataset.

#### Any other topics

* N/A

---

### Date: 05 February 2026

#### Who did you help this week?

N/A (This was the first supervision meeting).

#### What helped you this week?

The meeting with Jørgen and Veronika helped clarify the direction for the project.

#### What did you achieve?

* Agreed on Project Scope & Data:

  * Dataset: MIMIC-IV-ECG (approx. 800,000 ECGs, 160,000 patients).

  * Data specs: 12 leads, 10 seconds long, 500 Hz (5,000 points), WFDB format 16.

  * Source: Beth Israel Deaconess Medical Center (2008–2019).

* Methodology:

  * Build a CNN-based autoencoder to explore the latent space (e.g., to find AFib clusters).

  * Apply dimensionality reduction (UMAP/t-SNE) to visualize vectors (e.g., 64 dimensions to x/y coordinates).

  * Training will occur on Jørgen’s hardware.

* Practicalities:

  * Fixed days: ITU on Thursday afternoons, KU on Tuesdays.

  * Project Title: "Project of Anton - Unsupervised Deep Learning of ECGs: Exploring the Latent Space".

#### What did you struggle with?

* Uncertainty regarding the direction of the project and administrative access (seeing the thesis in LearnIT and getting access to KU's systems).

#### What would you like to work on next week?

* Call SAP to check LearnIT access, confirm title registration, and clarify problem statement deadlines.

  * Register supervisor with SAP using the title: "Project of Anton - Unsupervised Deep Learning of ECGs: Exploring the Latent Space".

* Set up a regular GitHub repository ([Template](https://github.com/drivendataorg/cookiecutter-data-science)) and create a WeeklyMeetings.md file.

* Set up Overleaf in a separate repository ([Template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn)).

* Create README and Requirements files.

* Draft the Problem Statement.

* Create a flowchart for Inclusion/Exclusion criteria and Methods.

* Review guidelines for writing.

* Check ECG file consistency and create a log-file (Task assigned by Jørgen 06/02).

#### Where do you need help from Veronika?

* Providing feedback for the Problem Statement.

#### Any other topics

* Availability: I have a 15-hour side job to balance with the thesis.

* Vacation: I have 2 weeks of vacation planned for March.

---

### Date: [Template]

#### Who did you help this week?

#### What helped you this week?

Replace this text with a one/two sentence description of what helped you this week and how.

#### What did you achieve?

* Replace this text with a bullet point list of what you achieved this week.
* It's ok if your list is only one bullet point long!

#### What did you struggle with?

* Replace this text with a bullet point list of where you struggled this week.
* It's ok if your list is only one bullet point long!

#### What would you like to work on next week?

* Replace this text with a bullet point list of what you would like to work on next week.
* It's ok if your list is only one bullet point long!
* Try to estimate how long each task will take.

#### Where do you need help from Veronika?

* Replace this text with a bullet point list of what you need help from Veronika on.
* It's ok if your list is only one bullet point long!
* Try to estimate how long each task will take.

#### Any other topics

This space is yours to add to as needed.