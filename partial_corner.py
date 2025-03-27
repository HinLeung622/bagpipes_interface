import numpy as np

from corner import hist2d,quantile
from matplotlib import pyplot as pl
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter


def partial_corner(
    xs,
    axes,
    bins=20,
    range=None,
    weights=None,
    color="k",
    hist_bin_factor=1,
    smooth=None,
    smooth1d=None,
    labels=None,
    label_kwargs=None,
    titles=None,
    show_titles=False,
    title_fmt=".2f",
    title_kwargs=None,
    truths=None,
    truth_color="#4682b4",
    scale_hist=False,
    quantiles=None,
    verbose=False,
    fig=None,
    max_n_ticks=5,
    top_ticks=False,
    use_math_text=False,
    reverse=False,
    labelpad=0.0,
    hist_kwargs=None,
    **hist2d_kwargs
):

    """ A slightly editted version of corner.corner, allowing for an array of matplotlib axes to be passed where the corner plots can be drawn, instead of only accepting the figure object in default corner"""
    
    if quantiles is None:
        quantiles = []
    if title_kwargs is None:
        title_kwargs = dict()
    if label_kwargs is None:
        label_kwargs = dict()


    # If no separate titles are set, copy the axis labels
    if titles is None:
        titles = labels

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], (
        "I don't believe that you want more " "dimensions than samples!"
    )

    # Parse the parameter ranges.
    # If any of the extents are percentiles, convert them to ranges.
    # Also make sure it's a normal list.
    range = list(range)
    for i, _ in enumerate(range):
        try:
            emin, emax = range[i]
        except TypeError:
            q = [0.5 - 0.5 * range[i], 0.5 + 0.5 * range[i]]
            range[i] = quantile(xs[i], q, weights=weights)

    if len(range) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in range]
    except TypeError:
        if len(bins) != len(range):
            raise ValueError("Dimension mismatch between bins and range")
    try:
        hist_bin_factor = [float(hist_bin_factor) for _ in range]
    except TypeError:
        if len(hist_bin_factor) != len(range):
            raise ValueError(
                "Dimension mismatch between hist_bin_factor and " "range"
            )

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0  # size of one side of one panel
    if reverse:
        lbdim = 0.2 * factor  # size of left/bottom margin
        trdim = 0.5 * factor  # size of top/right margin
    else:
        lbdim = 0.5 * factor  # size of left/bottom margin
        trdim = 0.2 * factor  # size of top/right margin
    whspace = 0.05  # w/hspace size
    plotdim = factor * K + factor * (K - 1.0) * whspace
    dim = lbdim + plotdim + trdim
    
    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = dict()
    hist_kwargs["color"] = hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")
        
    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            if reverse:
                ax = axes[K - i - 1, K - i - 1]
            else:
                ax = axes[i, i]
        # Plot the histograms.
        if smooth1d is None:
            bins_1d = int(max(1, np.round(hist_bin_factor[i] * bins[i])))
            n, _, _ = ax.hist(
                x,
                bins=bins_1d,
                weights=weights,
                range=np.sort(range[i]),
                **hist_kwargs
            )
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, b = np.histogram(
                x, bins=bins[i], weights=weights, range=np.sort(range[i])
            )
            n = gaussian_filter(n, smooth1d)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
            ax.plot(x0, y0, **hist_kwargs)

        if truths is not None and truths[i] is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=color)

            if verbose:
                print("Quantiles:")
                print([item for item in zip(quantiles, qvalues)])

        if show_titles:
            title = None
            if title_fmt is not None:
                # Compute the quantiles for the title. This might redo
                # unneeded computation but who cares.
                q_16, q_50, q_84 = quantile(
                    x, [0.16, 0.5, 0.84], weights=weights
                )
                q_m, q_p = q_50 - q_16, q_84 - q_50

                # Format the quantile display.
                fmt = "{{0:{0}}}".format(title_fmt).format
                title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))

                # Add in the column name if it's given.
                if titles is not None:
                    title = "{0} = {1}".format(titles[i], title)

            elif titles is not None:
                title = "{0}".format(titles[i])

            if title is not None:
                if reverse:
                    if "pad" in title_kwargs.keys():
                        title_kwargs_new = copy.copy(title_kwargs)
                        del title_kwargs_new["pad"]
                        title_kwargs_new["labelpad"] = title_kwargs["pad"]
                    else:
                        title_kwargs_new = title_kwargs

                    ax.set_xlabel(title, **title_kwargs_new)
                else:
                    ax.set_title(title, **title_kwargs)

        # Set up the axes.
        ax.set_xlim(range[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            if reverse:
                ax.xaxis.tick_top()
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                if reverse:
                    if "labelpad" in label_kwargs.keys():
                        label_kwargs_new = copy.copy(label_kwargs)
                        del label_kwargs_new["labelpad"]
                        label_kwargs_new["pad"] = label_kwargs["labelpad"]
                    else:
                        label_kwargs_new = label_kwargs
                    ax.set_title(
                        labels[i],
                        position=(0.5, 1.3 + labelpad),
                        **label_kwargs_new
                    )

                else:
                    ax.set_xlabel(labels[i], **label_kwargs)
                    ax.xaxis.set_label_coords(0.5, -0.3 - labelpad)

            # use MathText for axes ticks
            ax.xaxis.set_major_formatter(
                ScalarFormatter(useMathText=use_math_text)
            )

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                if reverse:
                    ax = axes[K - i - 1, K - j - 1]
                else:
                    ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif j == i:
                continue

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            hist2d(
                y,
                x,
                ax=ax,
                range=[range[j], range[i]],
                weights=weights,
                color=color,
                smooth=smooth,
                bins=[bins[j], bins[i]],
                **hist2d_kwargs
            )

            if truths is not None:
                if truths[i] is not None and truths[j] is not None:
                    ax.plot(truths[j], truths[i], "s", color=truth_color)
                if truths[j] is not None:
                    ax.axvline(truths[j], color=truth_color)
                if truths[i] is not None:
                    ax.axhline(truths[i], color=truth_color)

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(
                    MaxNLocator(max_n_ticks, prune="lower")
                )
                ax.yaxis.set_major_locator(
                    MaxNLocator(max_n_ticks, prune="lower")
                )

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                if reverse:
                    ax.xaxis.tick_top()
                [l.set_rotation(45) for l in ax.get_xticklabels()]
                if labels is not None:
                    ax.set_xlabel(labels[j], **label_kwargs)
                    if reverse:
                        ax.xaxis.set_label_coords(0.5, 1.4 + labelpad)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.3 - labelpad)

                # use MathText for axes ticks
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text)
                )

            if j > 0:
                ax.set_yticklabels([])
            else:
                if reverse:
                    ax.yaxis.tick_right()
                [l.set_rotation(45) for l in ax.get_yticklabels()]
                if labels is not None:
                    if reverse:
                        ax.set_ylabel(labels[i], rotation=-90, **label_kwargs)
                        ax.yaxis.set_label_coords(1.3 + labelpad, 0.5)
                    else:
                        ax.set_ylabel(labels[i], **label_kwargs)
                        ax.yaxis.set_label_coords(-0.3 - labelpad, 0.5)

                # use MathText for axes ticks
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text)
                )

    return fig
