# Tuto 4: Analyze Preprocessed Signal

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">


## 1️⃣ Visualize preprocessed signal
See [Tuto](raw-viz.md)

## 2️⃣ Analyze Your Signal Manually
In the <i class="bi bi-activity"></i> **Analyze** tab:

- **Activate the Topomap**: click once on the button to activate the feature (click again to deactivate). Then click on the signal at the desired timepoint to view spatial maps.
- **Add annotations**: click or select a segment on the graph to auto-fill onset and duration (set duration = 0 for punctual events). Add a label and click **Add New Event**. It will appear in the annotation options (make sure it is selected to be visible on the graph).
- **Delete annotations**: select a segment and click **Deleted Selected Event(s)** to remove the current event(s) in the selection.
- **To delete an entire event category**, go to the **Select** tab, choose the event type, and click **Delete**, then confirm.

## 3️⃣ Run a Prediction Model

Available MEG models :

|        Model       | Window length |      Input data     |  Framework | Number of parameters | Execution time (s)* | Performance (F1-score)** |
|:------------------:|:-------------:|:-------------------:|:----------:|:--------------------:|:-------------------:|:------------------------:|
|         CNN        |     200ms     |    Raw MEG signal   | Tensorflow |       1,095,681      |          53         |           0.471          |
| Feature- based ANN |     200ms     | 6 features/ channel | Tensorflow |        913,000       |          26         |           0.400          |
|     Transformer    |     400ms     |    Raw MEG signal   |   Pytorch  |       3,329,282      |          123        |           0.351          |

*: For 3-minutes MEG recording with 275 sensors. Using the Docker install, running on a GPU (NVIDIA RTX A2000 8GB Laptop GPU)
**: Results averaged across 10 patients from the Lyon dataset (see publication).


Available EEG models :

|        Model       | Window length |      Input data     |  Framework | Number of parameters | Execution time (s)* | Performance (F1-score)** |
|:------------------:|:-------------:|:-------------------:|:----------:|:--------------------:|:-------------------:|:------------------------:|
|      UIEDNET       |       3s      |    Raw EEG signal   |   Pytorch  |       2,960,195      |          7          |           0.551          |
|     Transformer    |     200ms     |    Raw EEG signal   |   Pytorch  |        819,714       |          24         |           0.435          |
|       HBIOT        |       2s      |    Raw EEG signal   |   Pytorch  |        603,266       |          12         |           0.482          |

*: For 20-minutes EEG recording with 19 sensors. Using the Docker install, running on a GPU (NVIDIA RTX A2000 8GB Laptop GPU)
**: Results averaged across 40 patients from the Lyon dataset and [vEpi](https://www.vepiset.com/) public dataset.

> Full training details for every pretrained model (modality, expected input,
> training preprocessing, reference) are listed on the **📊 Model Performance** page.

In the <i class="bi bi-stars"></i> **Spike Prediction** tab:

- **Select your model**.
- **Select the signal to use**.
- **Optional**: enable **sensitivity analysis** and **onset adjustement**.
    - Sensitivity analysis (SmoothGrad) averages gradients over noisy inputs to highlight the most influential regions of the signal (available for CNN MEG model).
    - You can also choose GFP-based alignment, which adjusts the spike onset to the peak of Global Field Power for greater accuracy.
- Click **Run Prediction**. It should complete in under a minute (on GPU).
- When results appear:
    - **Adjust the threshold** to refine spike detection based on the displayed probability distribution.
    - **Rename the result** (default: model_name_threshold_value) and click **Save** to make it selectable in the graph view.

> If SmoothGrad was selected, a new color layer will be added in the **Select** panel. It highlights predicted spike regions on the signal.

> Running the same model again for the same subject will return cached results instantly—you can re-save with a different threshold.
