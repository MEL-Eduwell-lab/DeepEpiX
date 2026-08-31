# Tuto 1 : Load & Preprocess Data

![Preprocessing Steps](../images/preprocessing.png)

## 1️⃣ Go to Home page

Open the app as explained in the **Setup & Installation** section.
You should arrive at the 🏠 **Home** page.  
If not, use the **☰ menu** at the top left to navigate there.

## 2️⃣ Choose a Subject

Use the **🔽 dropdown** menu to select a subject.  
> *Note:* The **📂 Open Folder** button only works if you have installed DeepEpiX locally.

You can open the following types of files:

- `.ds` folders (CTF MEG)
- `.fif` files (MEG/EEG)
- `.edf` / `.bdf` files (EEG)
- `.vhdr` files (BrainVision EEG)
- `.set` files (EEGLAB EEG)
- `4D` folders (must include at least: `rfDC-EEG`, `config` and `hs-file`)

The modality (MEG or scalp EEG) is detected automatically from the loaded file
and drives which preprocessing options and prediction models are available.

## 3️⃣ Load and Access Metadata

When you click on **📥 Load**, the previous database memory is cleared and **⚙️ preprocessing parameters** become accessible.  
You can adjust these parameters while exploring:

- Metadata (`raw.info`)
- Past annotations 
- Power Spectrum Decomposition (as a function of frequency parameters)

## 4️⃣ Preprocess and Visualize

In this step, the following preprocessing operations are applied:

- Resampling
- Band-pass filtering
- Notch filtering (to remove line noise)

---

### Optional Settings

- **💓 ECG Channel Detection**

    Specify a channel that clearly captures the heartbeat for ECG event detection using ```mne.find_ecg_events```.
    Default: ```None``` (all channels are used)

- **🧠 EEG Reference** *(scalp EEG recordings only)*

    Re-reference EEG channels before filtering.
    Default: ```Average```.
    Choosing ```Bipolar (double banana)``` applies the standard temporal
    "double banana" montage; pairs whose electrodes are missing from the
    recording are skipped.
    This option is disabled for MEG recordings, which don't need re-referencing.

- **❌ Bad Channels**

    Specify channels to exclude from analysis (e.g., topographic plots, model predictions), while still allowing them to be visualized. 
    These channels will be marked as bad and grouped accordingly.
    Accepts a single channel name or a comma-separated list. 
    These will be added to the existing bad channels in ```raw.info```.
    If desired, they can be saved with the new annotations at the end of your session.


> *Tip:* To find the correct channel name format, check section **🗂️ Raw Info** → **Channel Names**.

---

Clicking **⚡ Preprocess** will:

- Filter and resample the data
- Store it in memory for the session duration
- Take you to the **📈 Raw Visualization** main page.

To view metadata again, return to the **🏠 Home** page and check the **📚 Database** table.

![Database Memory](../images/database.png)
