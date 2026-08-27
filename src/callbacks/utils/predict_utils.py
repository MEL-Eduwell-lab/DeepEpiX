from pathlib import Path
from config import MODELS_DIR, CACHE_DIR
import os
import dask.dataframe as dd
import pandas as pd

from callbacks.utils import preprocessing_utils as pu


# Models are stored in per-modality sub-folders of MODELS_DIR.
MODALITY_MODEL_DIRS = {
    "meg": ["MEG"],
    "eeg": ["EEG"],
    "mixed": ["MEG", "EEG"],
}


def get_model_options(modality=None):
    """Build the dropdown options for the models matching a given modality.

    Parameters
    ----------
    modality : str or None
        Recording modality ('meg', 'eeg', 'mixed'). When unknown or None,
        every available model is listed.
    """
    model_dir = Path(MODELS_DIR)
    subdirs = MODALITY_MODEL_DIRS.get((modality or "").lower(), ["MEG", "EEG"])

    options = []
    for sub in subdirs:
        sub_path = model_dir / sub
        if not sub_path.is_dir():
            continue
        for f in sorted(sub_path.iterdir()):
            if f.is_file():
                options.append(
                    {"label": f"{sub} · {f.name}", "value": str(f.resolve())}
                )

    if not options:
        return [{"label": "No model available", "value": ""}]
    return options

def extract_preprocess_signal(
    data_path,
    freq_data,
    channels_dict,
    chunk_limits,
    excluded_ica_components=None,
    cache_dir=f"{CACHE_DIR}",
) -> pd.DataFrame:
    """
    Reconstruct the full preprocessed signal from cached Parquet segments.

    This function identifies missing segments based on provided chunk limits,
    triggers a preprocessing fallback for any missing data, and then
    concatenates all segments into a single pandas DataFrame.

    Parameters
    ----------
    data_path : str
        Path to the source M/EEG file.
    freq_data : dict
        Filtering and sampling parameters. Should contain keys like 
        'resample_freq' and 'low_pass_freq'.
    channels_dict : dict
        Mapping of channel groups to their respective names/indices.
    chunk_limits : list of list of float
        A list of [start, end] time boundaries (in seconds) used to 
        identify required cache segments.
    excluded_ica_components : list of int, optional
        Indices of ICA components to be removed from the signal. 
        Defaults to None.
    cache_dir : str, optional
        Directory where Parquet files are stored. Defaults to 
        `config.CACHE_DIR`.

    Returns
    -------
    pd.DataFrame
        The complete reconstructed signal, indexed by time.

    Notes
    -----
    The function uses Dask to read Parquet files in parallel before 
    calling `.compute()` to return a standard Pandas DataFrame. Ensure 
    system memory is sufficient for the total signal size.
    """
    os.makedirs(cache_dir, exist_ok=True)

    found_segments, missing_segments = _find_cached_segments(
            data_path, freq_data, excluded_ica_components,
            cache_dir, chunk_limits
        )

    if missing_segments:
        prep_raw_obj = pu.sort_filter_resample(data_path, freq_data, channels_dict)

        for chunk in missing_segments:
            pu.get_preprocessed_dataframe_dask(
                data_path=data_path,
                freq_data=freq_data,
                start_time=chunk[0],
                end_time=chunk[1],
                channels_dict=channels_dict,
                excluded_ica_components=excluded_ica_components,
                cache_dir=cache_dir,
                prep_raw=prep_raw_obj,
            )

        found_segments, _ = _find_cached_segments(
            data_path, freq_data, excluded_ica_components,
            cache_dir, chunk_limits
        )

    ddfs = [dd.read_parquet(seg_path) for seg_path in found_segments] #type: ignore
    return dd.concat(ddfs).compute() #type: ignore
    
def _find_cached_segments(
    data_path, freq_data, excluded_ica_components,
    cache_dir, chunk_limits 
):
    """
    Identify which required data segments are already available in the cache.

    Iterates through the provided time chunks and checks for the existence 
    of corresponding Parquet files using a standardized naming convention.

    Parameters
    ----------
    data_path : str
        Path to the source M/EEG file.
    freq_data : dict
        Filtering and sampling parameters used for generating the cache filename.
    excluded_ica_components : list of int or None
        ICA components excluded during preprocessing, used for cache lookup.
    cache_dir : str
        Directory to search for Parquet files.
    chunk_limits : list of list of float
        List of [start, end] time intervals to check.

    Returns
    -------
    found_segments : list of str
        List of file paths for segments found in the cache, sorted by start time.
    missing_segments : list of list of float
        List of [start, end] chunks that were not found in the cache.
    """
    found_segments = []
    missing_segments = []

    for chunk in sorted(chunk_limits, key=lambda x: x[0]):
        start, end = chunk[0], chunk[1]

        candidate = pu.get_cache_filename(
            data_path, freq_data, start, end,
            cache_dir, excluded_ica_components or None
        )

        if os.path.exists(candidate):
            found_segments.append((start, candidate))
        else:
            missing_segments.append(chunk)

    return (
        [path for _, path in found_segments],
        missing_segments
    )   