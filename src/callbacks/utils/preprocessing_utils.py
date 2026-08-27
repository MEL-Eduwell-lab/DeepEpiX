# Dash & Plotly
import dash
from dash import dcc
import plotly.graph_objects as go

# Standard Library
import os
import hashlib
from pathlib import Path
import json

# Third-party Libraries
import numpy as np
import mne
import dask.dataframe as dd
from dask.delayed import delayed
from flask_caching import Cache

# Local Modules
import config
from callbacks.utils import path_utils as dpu
from callbacks.utils import cache_utils as cu
from callbacks.utils import channel_utils as chu

app = dash.get_app()

cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": config.CACHE_TYPE,
        "CACHE_DIR": config.CACHE_DIR,
        "CACHE_DEFAULT_TIMEOUT": config.CACHE_DEFAULT_TIMEOUT,
    },
)

# RAW DATA PREPROCESSING (FILTERING, SUBSAMPLING) #################################################################


def get_notch_harmonics(notch_freq, sfreq):
    """
    Expand a base powerline notch frequency into its harmonics (2x, 3x),
    e.g. 50 -> [50, 100, 150], keeping only harmonics below the Nyquist
    frequency (sfreq / 2).
    """
    if not notch_freq:
        return notch_freq
    nyquist = sfreq / 2
    harmonics = [h for h in (notch_freq, notch_freq * 2, notch_freq * 3) if h < nyquist]
    return harmonics or None


def get_original_signal_freq_data(freq_data):
    """
    Variant of freq_data for computations that must run on the original
    scalp channel space -- ICA fitting/application and topomaps -- rather
    than the bipolar montage used for display. Bipolar re-referencing
    replaces scalp channels with derived anode-cathode pairs, which have no
    direct correspondence to electrode positions or to an ICA fit on the
    original channels. "average" reference is left untouched since it
    doesn't change channel identity.
    """
    if freq_data.get("reference") == "double_banana_bipolar":
        return {**freq_data, "reference": "none"}
    return freq_data


def sort_filter_resample(data_path, freq_data, channels_dict):

    raw = dpu.read_raw(data_path, preload=True, verbose=False)

    channels_order = [ch for group in channels_dict.values() for ch in group]
    raw.reorder_channels(channels_order)

    resample_freq = freq_data.get("resample_freq")
    low_pass_freq = freq_data.get("low_pass_freq")
    high_pass_freq = freq_data.get("high_pass_freq")
    notch_freq = freq_data.get("notch_freq")
    reference = freq_data.get("reference")

    if reference == "average":
        scalp_names = [raw.ch_names[i] for i in chu.get_scalp_eeg_picks(raw.info)]
        if scalp_names:
            raw.set_eeg_reference(ref_channels=scalp_names)
    elif reference == "double_banana_bipolar":
        raw = chu.apply_bipolar_montage(raw)

    # Apply filtering and resampling
    raw.filter(l_freq=high_pass_freq, h_freq=low_pass_freq, n_jobs=-1)
    if notch_freq:
        raw.notch_filter(freqs=notch_freq, n_jobs=-1)
    raw.resample(resample_freq, n_jobs=-1)

    # raw.rename_channels({ch['ch_name']: ch['ch_name'].split('-')[0] for ch in raw.info['chs']})

    return raw


# MAIN CACHED FUNCTIONS (PREPROCESSING) ##############################################


def get_max_length(raw, resample_freq):
    return raw.times[-1] - 1 / resample_freq


def update_chunk_limits(total_duration):
    chunk_duration = config.CHUNK_RECORDING_DURATION
    chunk_limits = [
        [start, min(start + chunk_duration, total_duration)]
        for start in range(0, int(total_duration), chunk_duration)
    ]
    return chunk_limits


def get_cache_filename(
    data_path, freq_data, start_time=None, end_time=None, 
    cache_dir=f"{config.CACHE_DIR}", excluded_components=None
):
    import json
    # Create a unique hash key
    hash_input = f"{data_path}_{json.dumps(freq_data, sort_keys=True)}"

    if start_time is not None and end_time is not None:
        hash_input += f"_{start_time}_{end_time}"

    if excluded_components is not None:
        hash_input += f"_ica_excluded_{sorted(excluded_components)}"

    unique_id = hashlib.md5(hash_input.encode()).hexdigest()
    filename = f"cache_{unique_id}.parquet"
    return os.path.join(cache_dir, filename)

def save_mne_sidecar(cache_file, prep_raw):
    """
    Save MNE metadata associated with a parquet file as a json file
    with same name.

    Parameters
    ----------
    cache_file : str
        Path to the reference parquet file
    prep_raw : mne.io.Raw or dask.delayed
        Raw MNE Object to extract metadata.
    """

    meta_path = cache_file.replace(".parquet", "_mne_meta.json")
    if os.path.exists(meta_path):
        return  # déjà sauvegardé

    try:
        raw = prep_raw.compute() if hasattr(prep_raw, "compute") else prep_raw
    except Exception:
        return

    channel_types = {
        ch["ch_name"]: mne.channel_type(raw.info, i)
        for i, ch in enumerate(raw.info["chs"])
    }

    meta = {
        "sfreq": raw.info["sfreq"],
        "highpass": raw.info["highpass"],
        "lowpass": raw.info["lowpass"],
        "channel_types": channel_types,         # {"MEG001": "mag", ...}
        "ch_names": raw.info["ch_names"],        # channel's order
        "bads": raw.info["bads"],

    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def get_preprocessed_dataframe_dask(
    data_path,
    freq_data,
    start_time,
    end_time,
    channels_dict,
    excluded_ica_components=None,
    prep_raw=None,
    cache_dir=f"{config.CACHE_DIR}",
):
    """
    Retrieve or compute a preprocessed Dask DataFrame for a specific time chunk.

    This function implements a disk-based caching strategy. It first checks if a 
    processed version of the requested data (defined by path, filters, and time) 
    already exists in Parquet format. If not, it executes a delayed computation 
    pipeline using Dask, standardizes the signal, and persists the result to disk.

    Parameters
    ----------
    data_path : str
        Path to the source M/EEG data file.
    freq_data : dict
        Dictionary containing filter parameters: 'resample_freq', 'low_pass_freq', 
        'high_pass_freq', and 'notch_freq'.
    start_time : float
        The start timestamp (in seconds) for the data crop.
    end_time : float
        The end timestamp (in seconds) for the data crop.
    channels_dict : dict
        Dictionary mapping channel groups (e.g., 'grad', 'mag') to channel names.
    excluded_ica_components : list of int, optional
        Indices of ICA components to be removed from the signal.
    prep_raw : mne.io.Raw, optional
        An existing MNE Raw object already filtered/resampled. If None, the 
        function will load and process the data from `data_path`.
    cache_dir : str, optional
        Directory where Parquet cache files are stored. Defaults to config.CACHE_DIR.

    Returns
    -------
    ddf : dask.dataframe.core.DataFrame
        A Dask DataFrame containing the preprocessed, standardized signal 
        indexed by time.
    prep_raw : MNE.Raw or None
        Preprocessed data as MNE format.
    """
    os.makedirs(cache_dir, exist_ok=True)

    if excluded_ica_components:
        cleaned_cache = get_cache_filename(
            data_path, freq_data, start_time, end_time, 
            cache_dir, excluded_ica_components
        )

        if os.path.exists(cleaned_cache):
            return dd.read_parquet(cleaned_cache), None #type: ignore
        
    cache_file = get_cache_filename(
        data_path, freq_data, start_time, end_time, cache_dir
    )

    if os.path.exists(cache_file):
        return dd.read_parquet(cache_file), None #type: ignore

    # Otherwise, compute and save
    @delayed
    def load_sort_filter():
        return sort_filter_resample(data_path, freq_data, channels_dict)

    @delayed
    def crop_and_to_df(prep_raw, s, e):
        raw_chunk = prep_raw.copy().crop(tmin=s, tmax=e, include_tmax=False)
        df = raw_chunk.to_data_frame(picks="all", index="time")
        current_min = df.index.min()
        if abs(current_min - s) > 1e-5:
            new_index = np.round(df.index.values + s, 8)
            df.index = new_index
            
        return df

    """
    @delayed
    def standardize(raw_df):
        return raw_df - raw_df.mean(axis=0)
    """
    
    if prep_raw is None:
        prep_raw = load_sort_filter()

    raw_df = crop_and_to_df(prep_raw, start_time, end_time)
    #raw_df_std = standardize(raw_df)

    df = raw_df.compute()
    ddf = dd.from_pandas(df, npartitions=10) #type: ignore
    ddf.to_parquet(cache_file)

    return ddf, prep_raw


# ICA ########################################################################


def get_cache_filename_ica(data_path, start_time, end_time, ica_result_path, cache_dir):
    hash_input = f"{data_path}_{start_time}_{end_time}_{ica_result_path}"
    hash_key = hashlib.md5(hash_input.encode()).hexdigest()
    return os.path.join(cache_dir, f"{hash_key}.parquet")

def run_ica_processing(data_path, n_components, 
                       ica_method, max_iter, decim, 
                       channel_store, cache_dir, ica_store):
    """
    Perform Independent Component Analysis (ICA) on M/EEG data with caching.

    This function checks if an ICA decomposition with the specified parameters 
    already exists in the cache and session store. If not, it loads the raw data, 
    applies a necessary 1Hz high-pass filter, fits the ICA model, and exports 
    diagnostic component plots to disk.

    Parameters
    ----------
    data_path : str or Path
        Path to the raw M/EEG data file.
    n_components : int or float
        Number of principal components to utilize. If float, it represents the 
        fraction of variance to explain.
    ica_method : {'fastica', 'infomax', 'picard'}
        The ICA algorithm to employ for decomposition.
    max_iter : int
        Maximum number of iterations for the solver to reach convergence.
    decim : int
        Downsampling factor (decimation) to reduce computation time during fitting.
    channel_store : dict
        Dictionary containing metadata about channels, specifically 'bad' channels.
    cache_dir : str or Path
        Directory where the resulting .fif files and diagnostic plots are saved.
    ica_store : list of str
        List of previously computed ICA file paths in the current session.

    Returns
    -------
    ica_result_path : Path
        The file path to the saved ICA solution.
    components_dir : Path
        The directory path where component diagnostic images are stored.
    is_from_cache : bool
        True if the result was loaded from disk, False if newly computed.
    n_components_actual : int
        The actual number of components extracted (useful if n_components was a float).
    explained_variance : float
        The ratio of variance explained by the kept ICA components relative 
        to the total PCA variance.
    """
    ica_result_path = Path(cache_dir) / f"{n_components}_{ica_method}_{max_iter}_{decim}-ica.fif"
    components_dir = Path(cache_dir) / f"{n_components}_{ica_method}_{max_iter}_{decim}-ica-components"

    if ica_result_path.exists() and str(ica_result_path) in ica_store:
        ica = mne.preprocessing.read_ica(ica_result_path)
        explained_variance = np.sum(ica.pca_explained_variance_[:ica.n_components_]) / np.sum(ica.pca_explained_variance_)
        return ica_result_path, components_dir, True, ica.n_components_, explained_variance

    raw = dpu.read_raw(
        data_path,
        preload=True,
        verbose=False,
        bad_channels=channel_store.get("bad", []),
    )
    modality = dpu.get_raw_modality(raw)

    # Restrict EEG picks to channels with a known scalp position (see
    # chu.get_scalp_eeg_picks): fitting ICA on auxiliary channels (ECG, EMG,
    # markers...) mislabeled as "eeg" degrades the decomposition, and without
    # a montage, ica.plot_components() can't draw the component topographies.
    meg_picks = (
        mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, ref_meg=False)
        if modality in ("meg", "mixed")
        else []
    )
    eeg_picks = chu.get_scalp_eeg_picks(raw.info) if modality in ("eeg", "mixed") else []
    raw.pick([raw.ch_names[i] for i in list(meg_picks) + list(eeg_picks)])

    if modality in ("eeg", "mixed"):
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"))

    raw.filter(l_freq=1.0, h_freq=None)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=ica_method,
        max_iter=max_iter,
        random_state=97,
    )
    ica.fit(raw, decim=decim)
    ica.save(ica_result_path, overwrite=True)

    _save_ica_component_plots(ica, components_dir)

    total_var = np.sum(ica.pca_explained_variance_)
    explained_var_sum = np.sum(ica.pca_explained_variance_[:ica.n_components_])
    explained_variance = explained_var_sum / total_var

    return ica_result_path, components_dir, False, ica.n_components_, explained_variance

def _save_ica_component_plots(ica, components_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    components_dir.mkdir(parents=True, exist_ok=True)

    figs = ica.plot_components(show=False)
    if not isinstance(figs, list):
        figs = [figs]

    for i, fig in enumerate(figs):
        fig.savefig(components_dir / f"page_{i}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

def get_ica_components_dask(
    data_path,
    start_time,
    end_time,
    ica_result_path,
    raw=None,
    cache_dir=f"{config.CACHE_DIR}",
):
    """
    Extract, resample, and cache ICA source time courses for a data segment.

    This function loads an existing ICA solution (if existing),
    projects the raw data into source space, resamples the resulting 
    components to 300 Hz, and saves the result as a Parquet file for 
    fast retrieval.

    Parameters
    ----------
    data_path : str
        Path to the raw MEG data file.
    start_time : float
        Start time of the segment in seconds.
    end_time : float
        End time of the segment in seconds.
    ica_result_path : str
        Path to the pre-computed MNE-ICA solution file.
    raw : mne.io.Raw, optional
        An existing MNE Raw object. If None, data is loaded and filtered at 1Hz.
    cache_dir : str, optional
        Directory for Parquet cache storage.

    Returns
    -------
    dask.dataframe.DataFrame
        A zero-mean Dask DataFrame containing the ICA source time courses 
        resampled to 300 Hz.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = get_cache_filename_ica(
        data_path, start_time, end_time, ica_result_path, cache_dir
    )

    if os.path.exists(cache_file):
        return dd.read_parquet(cache_file) #type: ignore

    if raw is None:
        raw = dpu.read_raw(data_path, preload=True, verbose=False)
        modality = dpu.get_raw_modality(raw)
        raw.pick(["meg", "eeg"] if modality == "mixed" else [modality])
        raw.filter(l_freq=1.0, h_freq=None)
    else:
        modality = dpu.get_raw_modality(raw)

    raw = raw.copy().crop(tmin=start_time, tmax=end_time)
    raw.pick(["meg", "eeg"] if modality == "mixed" else [modality])

    ica = mne.preprocessing.read_ica(ica_result_path)
    sources = ica.get_sources(raw)

    data = sources.get_data()
    sfreq = sources.info["sfreq"]
    ch_names = sources.ch_names
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="misc")

    new_sources = mne.io.RawArray(data, info)
    sources = new_sources.resample(300)
    sources_df = sources.to_data_frame(index="time") #type: ignore

    ddf = dd.from_pandas(sources_df, npartitions=10) #type: ignore
    ddf = ddf - ddf.mean(axis=0)
    ddf.to_parquet(cache_file)

    return ddf

def get_reconstructed_signal_dask(
        data_path,
        freq_data,
        start_time,
        end_time,
        ica_result_path,
        excluded_components,
        raw,
        cache_dir=f"{config.CACHE_DIR}",
    ):
    """
    Apply ICA artifact rejection and cache the reconstructed MEG signal.

    This function performs signal space projection using a pre-computed ICA 
    solution. It excludes specified artifactual components (e.g., eye blinks, 
    heartbeat), reconstructs the sensor-level signal, and persists the 
    result as a Parquet file for high-performance retrieval in the UI.

    Parameters
    ----------
    data_path : str
        Path to the raw MEG data file.
    freq_data : float
        Frequency information used for cache naming.
    start_time : float
        Start time of the data segment in seconds.
    end_time : float
        End time of the data segment in seconds.
    ica_result_path : str
        Path to the pre-computed MNE-ICA solution file.
    excluded_components : list of int
        Indices of ICA components to be removed (e.g., artifacts).
    raw : mne.io.Raw or None
        An existing MNE Raw object in the original scalp channel space
        (matching what ICA was fit on -- see run_ica_processing). If None,
        data is loaded from `data_path`.
    cache_dir : str, optional
        Directory to store the resulting Parquet files.

    Returns
    -------
    dask.dataframe.DataFrame
        The cleaned data represented as a partitioned Dask DataFrame.

    Notes
    -----
    ICA exclusion is applied only to the channel type(s) the ICA was fit on
    (MEG, EEG, or both for mixed recordings); other channels (stim, misc,
    or the complementary modality) are left at their original values.
    """
    os.makedirs(cache_dir, exist_ok=True)

    cache_file = get_cache_filename(
        data_path, freq_data, start_time, end_time,
        cache_dir, excluded_components
    )

    if os.path.exists(cache_file):
        return dd.read_parquet(cache_file) #type: ignore
    
    raw_chunk = raw.copy().crop(tmin=start_time, tmax=end_time)

    modality = dpu.get_raw_modality(raw_chunk)
    ica_meg = modality in ("meg", "mixed")
    ica_eeg = modality in ("eeg", "mixed")

    ica_picks = mne.pick_types(raw_chunk.info, meg=ica_meg, eeg=ica_eeg, stim=False, exclude=[])
    other_picks = mne.pick_types(raw_chunk.info, meg=not ica_meg, eeg=not ica_eeg, stim=True, misc=True, exclude=[])

    raw_ica_channels = raw_chunk.copy().pick(picks=ica_picks)
    if len(other_picks) > 0:
        raw_other_channels = raw_chunk.copy().pick(picks=other_picks)
    else:
        raw_other_channels = None

    ica = mne.preprocessing.read_ica(ica_result_path)
    ica.exclude = list(excluded_components)
    ica.apply(raw_ica_channels)

    if raw_other_channels is not None:
        cleaned_raw = raw_ica_channels.add_channels([raw_other_channels])
    else:
        cleaned_raw = raw_ica_channels

    # `raw` (and therefore `cleaned_raw`) is in the original scalp channel
    # space, since ICA is always fit/applied there (see run_ica_processing).
    # Re-derive the bipolar montage on the cleaned signal so the display
    # (which shows bipolar channels when that reference is selected)
    # reflects the ICA cleaning.
    if freq_data.get("reference") == "double_banana_bipolar":
        cleaned_raw = chu.apply_bipolar_montage(cleaned_raw)

    cleaned_df = cleaned_raw.to_data_frame(index="time")
    ddf = dd.from_pandas(cleaned_df, npartitions=10) #type: ignore
    ddf.to_parquet(cache_file, overwrite=True)

    return ddf


# POWER SPECTRUM DECOMPOSITION ######################################################


def compute_power_spectrum_decomposition(data_path, freq_data, theme="light"):
    raw = dpu.read_raw(data_path, preload=True, verbose=False)
    modality = dpu.get_raw_modality(raw)
    meg_picks = (
        mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False, ref_meg=False)
        if modality in ("meg", "mixed")
        else []
    )
    eeg_picks = chu.get_scalp_eeg_picks(raw.info) if modality in ("eeg", "mixed") else []

    picks = [raw.ch_names[i] for i in list(meg_picks) + list(eeg_picks)]

    low_pass_freq = freq_data.get("low_pass_freq")
    high_pass_freq = freq_data.get("high_pass_freq")
    notch_freq = freq_data.get("notch_freq")

    if not low_pass_freq or not high_pass_freq:
        return dash.no_update

    if notch_freq:
        raw.notch_filter(freqs=get_notch_harmonics(notch_freq, raw.info["sfreq"]))

    psd_data = raw.compute_psd(
        method="welch",
        fmin=high_pass_freq,
        fmax=low_pass_freq,
        n_fft=2048,
        picks=picks,
        n_jobs=-1,
    )
    psd, freqs = psd_data.get_data(return_freqs=True)

    psd_dB = 10 * np.log10(psd)

    text_color = "#000" if theme == "light" else "#FFF"
    grid_color = (
        "rgba(200,200,200,0.3)" if theme == "light" else "rgba(255,255,255,0.1)"
    )

    psd_fig = go.Figure()

    for ch_idx, ch_name in enumerate(psd_data.ch_names):
        psd_fig.add_trace(
            go.Scatter(
                x=freqs,
                y=psd_dB[ch_idx],
                mode="lines",
                line=dict(width=1),
                name=ch_name,
            )
        )

    psd_fig.update_layout(
        title=dict(text="Power Spectral Density (PSD)", font=dict(color=text_color)),
        xaxis=dict(
            title="Frequency (Hz)",
            type="linear",
            showgrid=True,
            gridcolor=grid_color,
            color=text_color,
        ),
        yaxis=dict(
            title="Power (dB)",
            type="linear",
            showgrid=True,
            gridcolor=grid_color,
            color=text_color,
        ),
        font=dict(color=text_color),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=text_color)),
    )

    return dcc.Graph(figure=psd_fig)

# PREDICTION ######################################################

def preprocess_same_as_training(model_config, model_name, data_path, channels_dict, cache_file):
    with open(model_config, 'r') as file:
        config = json.load(file)

    model_name = os.path.basename(model_name)
    model_cfg = config.get(model_name, {})

    notch = model_cfg.get("notch")
    freq_data = {
        "resample_freq": model_cfg.get("sfreq"),
        "low_pass_freq":  model_cfg.get("lowpass"),
        "high_pass_freq": model_cfg.get("highpass"),
        "notch_freq":     None if notch == "None" else notch,
    }
    
    @delayed
    def load_sort_filter():
        return sort_filter_resample(data_path, freq_data, channels_dict)

    @delayed
    def to_df(prep_raw):
        return prep_raw.to_data_frame(picks="all", index="time")

    prep_raw = load_sort_filter()
    raw_df = to_df(prep_raw)
    df = raw_df.compute()
    df.to_parquet(cache_file)

    save_mne_sidecar(cache_file, prep_raw)
    mne_info_path = cache_file.replace(".parquet", "_mne_meta.json")

    return mne_info_path, df