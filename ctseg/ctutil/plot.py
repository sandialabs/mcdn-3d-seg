"""plot.py
Utilities for visualizing data and predictions.

> Tyler Ganter, tganter@sandia.gov
"""
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np


def plot_img_or_slice(
    data=None,
    target=None,
    norm_data=None,
    data_recon=None,
    target_recon=None,
    pred=None,
    unc=None,
    nominal=None,
    fig_scale=6,
    overlay=False,
    alpha=0.5,
):
    """Plots a 2D image or slice of 3D volume, including original image, target label,
    prediction, etc.

    All inputs are optional and are excluded from the plot if not provided.

    Args:
        data (array_like): the input data to segment
        target (array_like): the binary target segmentation
        norm_data (array_like): the normalized input data
        data_recon (array_like): the reconstructed data (after chunking)
        target_recon (array_like): the reconstructed target (after chunking)
        pred (array_like): the predicted segmentation
        unc (array_like): the estimated prediction uncertainty
        nominal (array_like): ???
        fig_scale (int): a relative scalar that scales the overall figure size
        overlay (bool): whether or not to overlay the target and prediction on the data.
            If the prediction is not an integer (i.e. not thresholded to 0's and 1's)
            the prediction will not be overlayed regardless.
        alpha (float): the overlay transparency in range [0, 1]

    Returns:
        a figure handle
    """
    num_plots = sum(
        x is not None
        for x in (data, target, norm_data, data_recon, target_recon, pred, unc, nominal)
    )
    num_plots += int(target is not None and pred is not None)
    num_rows = int(num_plots ** 0.5)
    num_cols = int(np.ceil(num_plots / num_rows))

    fig = plt.figure(figsize=(fig_scale * num_cols, fig_scale * num_rows))
    cur_plot_no = 1

    # plot input data
    if data is not None:
        plt.subplot(num_rows, num_cols, cur_plot_no)
        data = np.array(data).squeeze()
        plt.imshow(data, cmap="Greys")
        plt.title("Data")
        frame = plt.gca()
        frame.axes.xaxis.set_visible(False)
        frame.axes.yaxis.set_visible(False)
        cur_plot_no += 1

    # plot target segmentation
    if target is not None:
        plt.subplot(num_rows, num_cols, cur_plot_no)
        target = np.array(target).squeeze()
        if overlay:
            plt.imshow(data, cmap="Greys")
            plt.imshow(-target, cmap="bwr", alpha=alpha * target)
        else:
            plt.imshow(target, cmap="Greys")
        plt.title("Target")
        frame = plt.gca()
        frame.axes.xaxis.set_visible(False)
        frame.axes.yaxis.set_visible(False)
        cur_plot_no += 1

    # plot normalized input data
    if norm_data is not None:
        plt.subplot(num_rows, num_cols, cur_plot_no)
        norm_data = np.array(norm_data).squeeze()
        plt.imshow(norm_data, cmap="Greys")
        plt.title("Normalized Data")
        frame = plt.gca()
        frame.axes.xaxis.set_visible(False)
        frame.axes.yaxis.set_visible(False)
        cur_plot_no += 1

    # plot reconstructed input data
    if data_recon is not None:
        plt.subplot(num_rows, num_cols, cur_plot_no)
        data_recon = np.array(data_recon).squeeze()
        plt.imshow(data_recon, cmap="Greys")
        plt.title("Reconstructed Data")
        frame = plt.gca()
        frame.axes.xaxis.set_visible(False)
        frame.axes.yaxis.set_visible(False)
        cur_plot_no += 1

    # plot reconstructed target segmentation
    if target_recon is not None:
        plt.subplot(num_rows, num_cols, cur_plot_no)
        target_recon = np.array(target_recon).squeeze()
        if overlay:
            plt.imshow(data, cmap="Greys")
            plt.imshow(-target_recon, cmap="bwr", alpha=alpha * target)
        else:
            plt.imshow(target_recon, cmap="Greys")
        plt.title("Reconstructed Target")
        frame = plt.gca()
        frame.axes.xaxis.set_visible(False)
        frame.axes.yaxis.set_visible(False)
        cur_plot_no += 1

    # plot predicted segmentation
    if pred is not None:
        pred = np.array(pred).squeeze()
        # is it a thresholded prediction?
        is_int_pred = np.issubdtype(pred.dtype, np.integer) or pred.dtype == np.bool

        plt.subplot(num_rows, num_cols, cur_plot_no)
        if overlay and is_int_pred:
            plt.imshow(data, cmap="Greys")
            plt.imshow(pred, cmap="bwr", alpha=alpha * pred)
        else:
            plt.imshow(pred, cmap="Greys")
        plt.title("Prediction")
        frame = plt.gca()
        frame.axes.xaxis.set_visible(False)
        frame.axes.yaxis.set_visible(False)
        cur_plot_no += 1

        # plot diff (prediction - target)
        if target is not None:
            plt.subplot(num_rows, num_cols, cur_plot_no)
            diff = pred - target
            if overlay and is_int_pred:
                plt.imshow(data, cmap="Greys")
                mask = diff != 0
                plt.imshow(diff, cmap="bwr", vmin=-1, vmax=1, alpha=alpha * mask)
            else:
                plt.imshow(diff, cmap="bwr", vmin=-1, vmax=1)
                plt.colorbar()
            plt.title("Diff (Pred - Target)")
            frame = plt.gca()
            frame.axes.xaxis.set_visible(False)
            frame.axes.yaxis.set_visible(False)
            cur_plot_no += 1

    # plot uncertainty
    if unc is not None:
        plt.subplot(num_rows, num_cols, cur_plot_no)
        unc = np.array(unc).squeeze()
        plt.imshow(unc, cmap=cc.cm.CET_L19)
        plt.colorbar()
        plt.title("Uncertainty")
        frame = plt.gca()
        frame.axes.xaxis.set_visible(False)
        frame.axes.yaxis.set_visible(False)
        cur_plot_no += 1

    # plot nominal
    if nominal is not None:
        plt.subplot(num_rows, num_cols, cur_plot_no)
        nominal = np.array(nominal).squeeze()
        plt.imshow(nominal, cmap="Greys")
        plt.title("Nominal")
        frame = plt.gca()
        frame.axes.xaxis.set_visible(False)
        frame.axes.yaxis.set_visible(False)

    fig.tight_layout()

    return fig


def plot_slices(
    data, num_slices_per_axis=3, axis=-1, include_edges=False, cmap="Greys", fig_scale=6
):
    """Plots multiple slices of a 3D volume

    Args:
        data (array-like): the 3D data
        num_slices_per_axis: number of slices to plot per axis
        axis: the axis to plot (axis=-1 corresponds to plotting all 3)
        include_edges: if True, include the edge indices (0 and -1) when choosing slices
        cmap: the colormap to plot with
        fig_scale (int): a relative scalar that scales the overall figure size

    Returns:
        a figure handle
    """
    data = np.array(data).squeeze()
    assert len(data.shape) == 3, f"Data is not 3D, it's {len(data.shape)}D"

    def make_slice(slice_axis, slice_idx):
        slc = [slice(None)] * data.ndim
        slc[slice_axis] = slice_idx
        return tuple(slc)

    slice_axis_idx = _make_even_slices(data, num_slices_per_axis, axis, include_edges)

    num_plots = len(slice_axis_idx)
    num_rows = int(num_plots ** 0.5)
    num_cols = int(np.ceil(num_plots / num_rows))

    fig = plt.figure(figsize=(fig_scale * num_cols, fig_scale * num_rows))
    cur_plot_no = 1

    for slice_axis, slice_idx in slice_axis_idx:
        slc = make_slice(slice_axis, slice_idx)
        cur_data = data[slc].squeeze()

        plt.subplot(num_rows, num_cols, cur_plot_no)
        plt.imshow(cur_data, cmap=cmap)
        plt.title(f"SliceAxis:{slice_axis} SliceIdx:{slice_idx}")
        frame = plt.gca()
        frame.axes.xaxis.set_visible(False)
        frame.axes.yaxis.set_visible(False)
        cur_plot_no += 1

    fig.tight_layout()

    return fig


def _make_even_slices(data, num_slices_per_axis, axis, include_edges):
    """Creates a list of evenly spaced slices

    Returns:
        a list of (slice_axis, slice_index) tuples
    """
    slices = []

    if axis == -1:
        # use all axis
        axes = range(3)
    else:
        axes = [axis]

    for cur_axis in axes:
        cur_shape = data.shape[cur_axis]
        if include_edges:
            step = cur_shape // (num_slices_per_axis - 1)
            slices += [
                (cur_axis, min(step * i, cur_shape - 1))
                for i in range(num_slices_per_axis)
            ]
        else:
            step = cur_shape // (num_slices_per_axis + 1)
            slices += [
                (cur_axis, min(step * (i + 1), cur_shape - 1))
                for i in range(num_slices_per_axis)
            ]

    return slices
