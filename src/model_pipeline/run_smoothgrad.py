import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
from scipy import signal
import tensorflow as tf
import tensorflow.keras as keras
import model_pipeline.utils as utils
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Computes the SmoothGrad input (repeat nb_repeat_sg times the original signal while adding noise)
def generate_noisy_input(f, win_idx, nb_repeat_sg, noise_val, dim):
    """
    Load a window and generate noisy copies for SmoothGrad inference.

    Parameters
    ----------
    f : file object
        Opened binary file containing MEG windows in float32 format.
    win_idx : int
        Index of the window to retrieve.
    nb_repeat_sg : int
        Number of noisy copies to generate for SmoothGrad.
    noise_val : float
        Standard deviation of the Gaussian noise to add.
    dim : tuple of int
        Shape of a single window as (n_channels, n_times).

    Returns
    -------
    sample_non_norm : numpy.ndarray
        Raw (non-normalized) window of shape (n_channels, n_times).
    noisy_images : numpy.ndarray
        Normalized window repeated and perturbed with Gaussian noise,
        shape (nb_repeat_sg, n_channels, n_times).
    """

    f.seek(dim[0] * dim[1] * win_idx * 4)
    sample_non_norm = np.fromfile(f, dtype="float32", count=dim[0] * dim[1])
    sample_non_norm = sample_non_norm.reshape(dim[1], dim[0])
    sample_non_norm = np.swapaxes(sample_non_norm, 0, 1)

    mean = np.mean(sample_non_norm)
    std = np.std(sample_non_norm)
    sample = (sample_non_norm - mean) / std
    sample = np.expand_dims(sample, 0)

    repeated_images = np.repeat(sample, nb_repeat_sg, axis=0)
    noise = np.random.normal(0, noise_val, repeated_images.shape).astype(np.float32)

    return sample_non_norm, repeated_images + noise


def get_av_grad(noisy_images, model, expected_output, num_samples):
    """
    Compute averaged SmoothGrad gradients and mean prediction over noisy inputs.

    Parameters
    ----------
    noisy_images : numpy.ndarray
        Noisy copies of the input window, shape (nb_repeat_sg, n_channels, n_times).
    model : tf.keras.Model
        Trained model used for inference and gradient computation.
    expected_output : numpy.ndarray
        Ground truth labels used to compute the loss.
    num_samples : int
        Number of noisy samples per original window, used to reshape gradients.

    Returns
    -------
    averaged_grads : tf.Tensor
        Absolute gradients averaged across noisy samples,
        shape (1, n_channels, n_times).
    mean_prediction : float
        Mean prediction score across all noisy inputs.
    """
    with tf.GradientTape() as tape:
        inputs = tf.cast(noisy_images, tf.float32)
        tape.watch(inputs)
        predictions = model(inputs)
        loss_func = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True, alpha=0.25, gamma=2
        )
        loss = loss_func(expected_output, predictions)
        
    grads = tape.gradient(loss, inputs)
    grads_per_image = tf.reshape(grads, (-1, num_samples, *grads.shape[1:]))
    averaged_grads = tf.reduce_mean(tf.abs(grads_per_image), axis=1)

    # Returns averaged gradient and averaged predictions over the SmoothGrad inputs
    return averaged_grads, np.mean(predictions.numpy())


def postprocess_grad(av_grad):
    """
    Postprocess averaged gradients by thresholding, rescaling, and smoothing.

    Values below the 75th percentile are suppressed to near-zero, while
    values above are rescaled to [0.75, 1.0]. A Wiener filter is then
    applied to smooth the result spatially.

    Parameters
    ----------
    av_grad : tf.Tensor
        Averaged absolute gradients of shape (1, n_channels, n_times).

    Returns
    -------
    numpy.ndarray
        Postprocessed gradient map of shape (n_channels, n_times).
    """
    av_grad_np = av_grad[0, :, :].numpy()

    thresh = np.quantile(av_grad_np, 0.75)
    above = av_grad_np > thresh
    av_grad_np[~above] = np.min(av_grad_np[above]) / 2

    av_grad_np[above] = 0.25 * (
        (av_grad_np[above] - np.min(av_grad_np[above]))
        / (np.max(av_grad_np[above]) - np.min(av_grad_np[above]))
    ) + 0.75

    return signal.wiener(av_grad_np, (7, 7))

# MAIN

def run_smoothgrad(
    model_file,
    model_type,
    path_to_files,
    y_pred_path,
    threshold,
    mne_info_cache_path,
    nb_repeat_sg,
    window_size,
    noise_val,
    total_length,
    overlap,
    signal_name,
):
    with open(mne_info_cache_path, "r") as f:
        metadata = json.load(f)

    sfreq = metadata['sfreq']
    dim = (int(sfreq * window_size), 275, 1)

    f = open(f"{path_to_files}/data_raw_windows_bi")
    blocks_file = utils.load_obj("data_raw_blocks.pkl", path_to_files)
    data_file = utils.load_obj("data_raw.pkl", path_to_files)
    full_result = pd.read_csv(y_pred_path)
    y_pred = full_result["probas"].to_numpy()

    total_nb_windows = len(blocks_file)
    total_nb_points = data_file["MEG/EEG"][0].shape[1]

    full_grads = np.zeros((total_nb_points, dim[1]))

    my_labels = np.ones((nb_repeat_sg, 1))
    dim = (int(sfreq * window_size), dim[1])

    # Model Selection
    if "TensorFlow" in model_type:
        # Get model
        model = keras.models.load_model(model_file, compile=False)

        # For each window
        for w, i in enumerate(range(0, total_nb_windows)):
            if y_pred[i] > threshold:

                # Noise Augmentation (generating multiple noisy copies of the input before averaging the gradients to reduce variance)
                sample_non_norm, noisy_images = generate_noisy_input(
                    f, w, nb_repeat_sg, noise_val, dim
                )

                # Compute SmoothGrad (average and normalizing)
                av_grad, pred = get_av_grad(
                    noisy_images, model, my_labels, nb_repeat_sg
                )

                norm_grads = postprocess_grad(av_grad)

                # If the model predicts a spike then fill the gradient array over the full window (comprising the beggining and end window overlaps)
                start = (w * total_length) - math.floor(overlap / 2)
                # fmt: off
                end = (w * total_length) + total_length + math.ceil(overlap / 2)
                # fmt : on

                full_grads[start:end, :] = norm_grads[:, :]

    signal_name = os.path.basename(signal_name).replace(".fif", "")
    grad_path = f"{path_to_files}/{os.path.basename(model_file)}_{signal_name}_smoothGrad.pkl"
    with open(grad_path, "wb") as f:
        pickle.dump(full_grads, f)


if __name__ == "__main__":
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    path_to_files = sys.argv[3]
    y_pred_path = sys.argv[4]
    threshold = float(sys.argv[5])  # Convert back to float
    mne_info_cache_path = sys.argv[6] if len(sys.argv) > 6 else None
    signal_name = sys.argv[7]

    # Parameters
    window_size = 0.2
    nb_repeat_sg = 10
    noise_val = 0.1
    centre_unique = 12
    overlap = 9
    total_lenght = centre_unique + overlap

    run_smoothgrad(
        model_path,
        model_type,
        path_to_files,
        y_pred_path,
        threshold,
        mne_info_cache_path,
        nb_repeat_sg,
        window_size,
        noise_val,
        total_lenght,
        overlap,
        signal_name,
    )
