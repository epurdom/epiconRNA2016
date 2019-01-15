import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colors import epicon_colors
import splines_


leaf_data = pd.read_csv("data/RNASeq/leaf_log.tsv",
                        delim_whitespace=True)
leaf_meta = pd.read_csv("data/RNASeq/leaf_meta.tsv",
                        delim_whitespace=True)

root_data = pd.read_csv("data/RNASeq/root_log.tsv",
                        delim_whitespace=True)
root_meta = pd.read_csv("data/RNASeq/root_meta.tsv",
                        delim_whitespace=True)


def plot_gene(gene_name, sample_type="leaf"):
    if not gene_name.endswith(".v3.1"):
        gene_name = gene_name + ".v3.1"

    if sample_type == "leaf":
        data = leaf_data
        meta = leaf_meta
    elif sample_type == "root":
        data = root_data
        meta = root_meta
    else:
        raise ValueError(
            "Unknown sample_type. Can be leaf or root. '%s' was "
            "provided." % sample_type)
    try:
        counts = data.loc[gene_name].values
    except KeyError:
        print("Gene not found in the data. Could be not expressed. '%s' was"
              " the gene name provided" % (gene_name,))
        return
    conditions = ["Preflowering", "Postflowering", "Control"]
    genotypes = ["RT430", "BT642"]

    fig, axes = plt.subplots(ncols=2, figsize=(12, 4),
                             gridspec_kw={"right": 0.7},
                             sharey=True)
    for i, geno in enumerate(genotypes):
        ax = axes[i]

        for i, condition in enumerate(conditions):
            mask = (
                (meta["Condition"] == condition) &
                (meta["Genotype"] == geno))
            sub_counts = counts[mask]
            sub_meta = meta[mask]
            sub_conditions = np.array(["%s.%s" % (c, g) for c, g in zip(
                sub_meta["Condition"].values,
                sub_meta["Genotype"].values)]).astype(object)

            timepoints = sub_meta["Week"].values.astype(int)

            if condition == "Preflowering":
                pre_timepoints = timepoints < 8.5
                sub_conditions[pre_timepoints] = [
                    "%s.pre" % c for c in sub_conditions[pre_timepoints]]
                df = 3
            elif "Postflowering" == condition:
                pre_timepoints = timepoints < 9.5
                sub_conditions[pre_timepoints] = [
                    "%s.pre" % c for c in sub_conditions[pre_timepoints]]
                df = 3
            else:
                df = 6

            for c in np.unique(sub_conditions):
                mask = sub_conditions == c
                if len(np.unique(timepoints[mask])) == 1:
                    ax.plot(np.unique(timepoints[mask]),
                            sub_counts[mask].mean(),
                            marker="o",
                            color=epicon_colors[condition])
                    continue
                basis = np.array(
                    splines_.get_basis_matrix(timepoints[mask],
                                              df=df,
                                              include_intercept=True))
                timepoints_prediction = np.linspace(timepoints[mask].min(),
                                                    timepoints[mask].max(),
                                                    100)
                basis_pred = np.array(
                    splines_.get_basis_matrix(
                        timepoints_prediction,
                        df=df, include_intercept=True))

                coef = splines_.estimate_splines_coefficient(
                    basis,
                    sub_counts[mask])
                splines_fitted = np.dot(coef, basis_pred.T)
                if "pre" in c:
                    label = ""
                else:
                    label = c.split(".")[0]
                    if label == "Preflowering":
                        label = "Pre-flowering"
                    elif label == "Postflowering":
                        label = "Post-flowering"
                ax.plot(timepoints_prediction,
                        splines_fitted,
                        color=epicon_colors[condition],
                        linewidth=3,
                        label=label)
                ax.plot(timepoints[mask],
                        sub_counts[mask],
                        color=epicon_colors[condition],
                        marker=".",
                        markersize=4,
                        linewidth=0)
                ax.set_xticks([3, 10, 17])
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.axvline(8.5, linestyle="--", color="0.5", zorder=-1)
        ax.axvline(9.5, linestyle="--", color="0.5", zorder=-1)

    axes[0].text(1, 1, "RTx430", fontweight="bold", fontsize="medium",
                 horizontalalignment="right",
                 transform=axes[0].transAxes)

    axes[1].text(1, 1, "BTx642", fontweight="bold", fontsize="medium",
                 horizontalalignment="right",
                 transform=axes[1].transAxes)
    axes[1].legend(bbox_to_anchor=(1.1, 1), frameon=False)
    axes[0].set_ylabel("Log expression", fontweight="bold")
    axes[0].set_xlabel("Week", fontweight="bold")
    axes[1].set_xlabel("Week", fontweight="bold")

    fig.suptitle("%s (%s)" % (gene_name, sample_type), fontsize="large",
                 fontweight="bold")
