from io import BytesIO
import base64
import matplotlib
import matplotlib.pyplot as plt
import mne

from callbacks.utils import channel_utils as chu

matplotlib.use("Agg")


def create_topomap_from_raw(raw, sfreq, t0, t):
    """
    Create a topomap image at a specific timepoint.
    Parameters:
    - raw: MNE Raw object
    - timepoint: Time in seconds
    Returns:
    - Base64-encoded string of the topomap image
    """

    # Extract the sampling frequency and calculate the time index
    timepoint = float(t - t0)
    time_idx = int(timepoint * sfreq)

    # Extract the data at the specified time index
    data = raw.get_data()  # Shape (n_channels, n_times)
    if time_idx < 0 or time_idx >= data.shape[1]:
        raise ValueError("Timepoint is out of range for the provided data.")

    mean_data = data[:, time_idx]

    fig, ax = plt.subplots()

    im, _ = mne.viz.plot_topomap(
        mean_data,
        raw.info,
        axes=ax,
        show=False,
        cmap="coolwarm",
        contours=6,
        sensors=True,  # Display the sensor locations
        res=128,
        sphere=None,
    )

    ax.axis("off")

    # Save the image to a buffer
    buf = BytesIO()
    fig.savefig(
        buf, format="png", bbox_inches="tight", pad_inches=0.1, transparent=True
    )
    buf.seek(0)

    # Encode the image in Base64
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    plt.close("all")

    return img_str


def create_topomap_from_preprocessed(
    original_raw, raw_ddf, sfreq, t0, t, bad_channels, modality
):
    """
    Create a topomap using preprocessed Dask data and original Raw metadata.

    Parameters:
    - raw_ddf: Dask DataFrame with shape (time, channels)
    - original_raw: MNE Raw object used to get info structure
    - timepoint: time in seconds at which to extract data

    Returns:
    - base64-encoded topomap image string
    """
    # Compute Dask DataFrame to NumPy
    preprocessed_df = raw_ddf.drop(columns=bad_channels).compute()

    # Dask DFs are usually (n_times, n_channels), so we transpose
    data = preprocessed_df.values.T

    info = original_raw.info.copy()

    if bad_channels:
        info["bads"] = bad_channels

    # Filter MEG channels if needed
    if modality == "meg":
        # Get only MEG channels (both magnetometers and gradiometers)
        picks = mne.pick_types(
            info,
            meg=True,
            eeg=False,
            stim=False,
            eog=False,
            ref_meg=False, #type: ignore
            exclude="bads",
        )

    elif modality == "eeg":
        picks = mne.pick_types(
            info, meg=False, eeg=True, stim=False, eog=False, exclude="bads"
        )

    elif modality == "mixed":
        # Get only MEG channels (both magnetometers and gradiometers)
        picks = mne.pick_types(
            info,
            meg=True,
            eeg=True,
            stim=False,
            eog=False,
            ref_meg=False, #type: ignore
            exclude="bads",
        )

    picked_info = mne.pick_info(info, picks)

    if modality in ("meg", "mixed"):
        raw_processed = mne.io.RawArray(data, picked_info).pick("mag")

    elif modality == "eeg":
        # Restrict to channels with a known position in the montage: auxiliary
        # physiological channels (ECG, EMG...) are often typed "eeg" by file
        # readers even though they aren't part of the scalp electrode layout,
        # and set_montage() would otherwise raise on their missing position.
        montage = mne.channels.make_standard_montage("standard_1020")
        scalp_idx = chu.get_scalp_eeg_picks(picked_info, montage=montage)
        scalp_info = mne.pick_info(picked_info, scalp_idx)
        scalp_info.set_montage(montage)
        raw_processed = mne.io.RawArray(data[scalp_idx], scalp_info)

    # Use your existing topomap function
    return create_topomap_from_raw(raw_processed, sfreq, t0, t)
