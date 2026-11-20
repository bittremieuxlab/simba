import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns





# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "/data/simba_files_"

# Pickle containing metadata-enhanced SIMBA results.
# Matches filenames containing "_unique_compounds".
METADATA_ENHANCED_SIMBA_PATTERN = os.path.join(
                DATA_DIR,
                #"analog_discovery_simba_modified_cosine_results_tf_unique_compounds.pkl",
                "analog_discovery_simba_modified_cosine_results_tf_simba_metadata_2.pkl",
)

# Modified Cosine results.
# Change ONLY this path when you want to use Modified Cosine data
# from a different pickle. This file is independent of both SIMBA files.
MODIFIED_COSINE_PICKLE = os.path.join(
                DATA_DIR,
                "analog_discovery_simba_modified_cosine_results_tf_unique_compounds.pkl",
)

# Pickle containing the original     no metadata SIMBA results.
NO_METADATA_SIMBA_PATTERN = os.path.join(
                DATA_DIR,
                "analog_discovery_simba_modified_cosine_results_tf_simba_no_metadata_2.pkl",
)

# MS2DeepScore results.
MS2DEEPSCORE_PICKLE = os.path.join(
                DATA_DIR,
                "analog_discovery_ms2deepscore_results_tf_unique_compounds.pkl",
)

# Ground-truth best MCES match.
GROUND_TRUTH_PICKLE = os.path.join(
                DATA_DIR,
                "analog_discovery_ground_truth.pkl",
)

# Output figure.
OUTPUT_PLOT = (
                "norm_mces_distance_"
                "best_match_modified_cosine_ms2deepscore_"
                "massspecgym_simba_metadata_enhanced_simba.png"
)

# Output pickle containing exactly the data used in the violin plot.
OUTPUT_PLOT_DATA_PICKLE = (
                "norm_mces_violin_plot_data.pkl"
)

CLIP_NORMALIZED_VALUES = False



# =============================================================================
# Algorithms to plot
# =============================================================================
#
# Choose any subset of these names.
# The order here is also the order in the violin plot.
#
# Available:
#      "Best match"
#      "Modified Cosine"
#      "MS2DeepScore"
#      "No metadata SIMBA"
#      "metadata enhanced SIMBA"
#
ALGORITHMS_TO_PLOT = [
                "Best match",
                "Modified Cosine",
                "MS2DeepScore",
                "No metadata SIMBA",
                "metadata enhanced SIMBA",
]







# =============================================================================
# File helpers
# =============================================================================

def find_single_pickle(pattern, description):
                """Find exactly one pickle matching a glob pattern."""
                matches = sorted(glob.glob(pattern))

                if len(matches) == 0:
                                raise FileNotFoundError(
                                                f"No pickle found for {description}.\n"
                                                f"Pattern: {pattern}"
                                )

                if len(matches) > 1:
                                raise RuntimeError(
                                                f"Multiple pickles found for {description}:\n"
                                                + "\n".join(matches)
                                )

                print(f"{description}: {matches[0]}")
                return matches[0]





METADATA_ENHANCED_SIMBA_PICKLE = None
NO_METADATA_SIMBA_PICKLE = None

# Only search for SIMBA pickle files that are actually needed.
# Modified Cosine now has its own independent MODIFIED_COSINE_PICKLE path.
if "metadata enhanced SIMBA" in ALGORITHMS_TO_PLOT:
                METADATA_ENHANCED_SIMBA_PICKLE = find_single_pickle(
                                METADATA_ENHANCED_SIMBA_PATTERN,
                                "Metadata-enhanced SIMBA pickle",
                )

if "No metadata SIMBA" in ALGORITHMS_TO_PLOT:
                NO_METADATA_SIMBA_PICKLE = find_single_pickle(
                                NO_METADATA_SIMBA_PATTERN,
                                "No metadata SIMBA pickle",
                )





# =============================================================================
# Data loading
# =============================================================================

def load_pickle_norm_mces_distances(pickle_file, method_name):
                """Load normalized MCES distances for one retrieval method."""

                if not os.path.exists(pickle_file):
                                raise FileNotFoundError(
                                                f"Pickle file not found: {pickle_file}"
                                )

                with open(pickle_file, "rb") as file:
                                results = pickle.load(file)

                key = "norm_mces_distances_by_method"

                if key not in results:
                                raise KeyError(
                                                f"{pickle_file!r} does not contain {key!r}. "
                                                f"Available top-level keys: {list(results.keys())}"
                                )

                distances_by_method = results[key]

                if method_name not in distances_by_method:
                                raise KeyError(
                                                f"Method {method_name!r} was not found in "
                                                f"{pickle_file!r}. "
                                                f"Available methods: "
                                                f"{list(distances_by_method.keys())}"
                                )

                values = np.asarray(
                                distances_by_method[method_name],
                                dtype=float,
                )

                # Remove NaN / inf values.
                return values[np.isfinite(values)]





def load_ground_truth_norm_mces_distances(pickle_file):
                """
                Load normalized MCES distances for the best possible MCES match.
                """

                if not os.path.exists(pickle_file):
                                raise FileNotFoundError(
                                                f"Ground-truth pickle not found: {pickle_file}"
                                )

                with open(pickle_file, "rb") as file:
                                results = pickle.load(file)

                key = "norm_mces_distances_by_method"

                if key not in results:
                                raise KeyError(
                                                f"Ground-truth pickle does not contain {key!r}. "
                                                f"Available top-level keys: {list(results.keys())}"
                                )

                distances_by_method = results[key]
                available_methods = list(distances_by_method.keys())

                print(
                                "\nGround-truth pickle methods:",
                                available_methods,
                )

                # Try different possible names for the ground-truth method.
                candidate_names = [
                                "ground_truth_mces",
                                "ground_truth",
                                "best_mces",
                                "best_mces_match",
                                "best_match",
                                "mces",
                ]

                selected_method = next(
                                (
                                                name
                                                for name in candidate_names
                                                if name in distances_by_method
                                ),
                                None,
                )

                if selected_method is None:

                                if len(available_methods) == 1:

                                                selected_method = available_methods[0]

                                                print(
                                                                "Automatically using the only method in "
                                                                f"ground-truth pickle: {selected_method!r}"
                                                )

                                else:

                                                raise KeyError(
                                                                "Could not automatically identify the "
                                                                "Best MCES method. "
                                                                f"Available methods: {available_methods}"
                                                )

                print(
                                f"Using ground-truth method: "
                                f"{selected_method!r}"
                )

                values = np.asarray(
                                distances_by_method[selected_method],
                                dtype=float,
                )

                values = values[np.isfinite(values)]

                print(
                                f"Ground truth loaded: "
                                f"n={len(values)}, "
                                f"median={np.median(values):.4f}, "
                                f"mean={np.mean(values):.4f}"
                )

                return values





# =============================================================================
# Method configuration
# =============================================================================
#
# Dictionary order determines the final violin-plot order:
#
# 1. Best match
# 2. Modified Cosine
# 3. MS2DeepScore
# 4. No metadata SIMBA
# 5. metadata enhanced SIMBA
#
# Modified Cosine is loaded from the independent MODIFIED_COSINE_PICKLE
# configured at the top of this script.
# =============================================================================

method_sources = {

                "Best match": {
                                "source_type": "ground_truth_pickle",
                                "file": GROUND_TRUTH_PICKLE,
                },

                "Modified Cosine": {
                                "source_type": "pickle",
                                "file": MODIFIED_COSINE_PICKLE,
                                "pickle_method": "modified_cosine",
                },

                "MS2DeepScore": {
                                "source_type": "pickle",
                                "file": MS2DEEPSCORE_PICKLE,
                                "pickle_method": "ms2deepscore",
                },

                "No metadata SIMBA": {
                                "source_type": "pickle",
                                "file": NO_METADATA_SIMBA_PICKLE,
                                "pickle_method": "simba",
                },

                "metadata enhanced SIMBA": {
                                "source_type": "pickle",
                                "file": METADATA_ENHANCED_SIMBA_PICKLE,
                                "pickle_method": "simba",
                },
}





def load_method_values(method_label, source_config):
                """Load values according to the source configuration."""

                source_type = source_config["source_type"]
                source_file = source_config["file"]

                print(f"Loading {method_label} from: {source_file}")

                if source_type == "ground_truth_pickle":

                                return load_ground_truth_norm_mces_distances(
                                                source_file
                                )

                if source_type == "pickle":

                                return load_pickle_norm_mces_distances(
                                                pickle_file=source_file,
                                                method_name=source_config["pickle_method"],
                                )

                raise ValueError(
                                f"Unsupported source type {source_type!r} "
                                f"for {method_label!r}."
                )





# =============================================================================
# Load data
# =============================================================================

plot_data = []
plot_labels = []

# Check the names before trying to load data.
unknown_algorithms = [
                name
                for name in ALGORITHMS_TO_PLOT
                if name not in method_sources
]

if unknown_algorithms:
                raise ValueError(
                                "Unknown algorithm(s) in ALGORITHMS_TO_PLOT: "
                                + ", ".join(repr(name) for name in unknown_algorithms)
                                + "\nAvailable algorithms: "
                                + ", ".join(method_sources.keys())
                )

if len(ALGORITHMS_TO_PLOT) == 0:
                raise ValueError(
                                "ALGORITHMS_TO_PLOT is empty. Select at least one algorithm."
                )

# Load only selected algorithms, in exactly the order specified above.
for method_label in ALGORITHMS_TO_PLOT:
                source_config = method_sources[method_label]

                try:

                                values = load_method_values(
                                                method_label,
                                                source_config,
                                )

                                # -------------------------------------------------------------
                                # Remove queries that could not be processed for ground truth.
                                #
                                # In this dataset, values >= 1 correspond to failed /
                                # non-processed ground-truth comparisons.
                                # -------------------------------------------------------------
                                if method_label == "Best match":

                                                values = np.asarray(
                                                                [
                                                                                value
                                                                                for value in values
                                                                                if value < 1
                                                                ],
                                                                dtype=float,
                                                )

                except (
                                FileNotFoundError,
                                KeyError,
                                TypeError,
                                ValueError,
                                RuntimeError,
                ) as error:

                                print(
                                                f"\nERROR loading {method_label}: "
                                                f"{error}"
                                )

                                continue

                if len(values) == 0:

                                print(
                                                f"Skipping {method_label}: "
                                                "no valid normalized MCES distances."
                                )

                                continue

                plot_data.append(values)
                plot_labels.append(method_label)

                print(
                                f"Loaded {method_label}: "
                                f"n={len(values)}, "
                                f"median={np.median(values):.4f}"
                )





if len(plot_data) == 0:

                raise ValueError(
                                "No valid normalized MCES distances were found."
                )





print("\nMethods that WILL be plotted:")

for label in plot_labels:
                print(f"     - {label}")





# =============================================================================
# Save violin plot data
# =============================================================================
#
# This saves the EXACT arrays that will be sent to matplotlib.
#
# Therefore:
#
# - NaN / inf values have already been removed.
# - Ground-truth values >= 1 have already been removed.
# - Method names correspond exactly to the labels in the figure.
#
# The pickle can later be loaded without needing any of the original
# SIMBA / MS2DeepScore / ground-truth files.
# =============================================================================

violin_plot_data = {
                label: np.asarray(values, dtype=float)
                for label, values in zip(
                                plot_labels,
                                plot_data,
                )
}

with open(
                OUTPUT_PLOT_DATA_PICKLE,
                "wb",
) as file:

                pickle.dump(
                                violin_plot_data,
                                file,
                                protocol=pickle.HIGHEST_PROTOCOL,
                )





print(
                f"\nSaved violin plot data to: "
                f"{OUTPUT_PLOT_DATA_PICKLE}"
)

print("\nSaved pickle contents:")

for method_label, values in violin_plot_data.items():

                print(
                                f"     {method_label}: "
                                f"n={len(values)}, "
                                f"median={np.median(values):.4f}, "
                                f"mean={np.mean(values):.4f}"
                )





# =============================================================================
# Plot
# =============================================================================
#
# Styling intentionally mirrors plotting_comparison_analog_discovery.ipynb:
# - seaborn "deep" palette
# - width-normalized violins with cut=0
# - transparent narrow boxplots overlaid on the violins
# - thick black whiskers / medians
# - stacked x-axis labels
# - soft grid and no title
#
# IMPORTANT: only the plotting style is changed here. plot_data is used exactly
# as loaded and saved above; no values are transformed, removed, or reordered.
# =============================================================================

# Same base palette used in the notebook.
colors_palette = sns.color_palette("deep", n_colors=6)

# Keep the notebook colors for methods that are shared with that figure.
# The metadata-enhanced model uses another color from the same "deep" palette.
method_colors = {
                "Best match": colors_palette[0],
                "Modified Cosine": colors_palette[1],
                "MS2DeepScore": colors_palette[4],
                "No metadata SIMBA": colors_palette[3],
                "metadata enhanced SIMBA": colors_palette[2],
}

violin_colors = [
                method_colors.get(label, colors_palette[i % len(colors_palette)])
                for i, label in enumerate(plot_labels)
]

fig, ax = plt.subplots(figsize=(6, 5))

# Violin plot: same appearance as the notebook comparison figure.
sns.violinplot(
                data=plot_data,
                inner=None,
                scale="width",
                cut=0,
                palette=violin_colors,
                ax=ax,
)

# Transparent boxplots overlaid on the violins.
sns.boxplot(
                data=plot_data,
                width=0.2,
                showcaps=True,
                boxprops={"facecolor": "none"},
                showfliers=False,
                whiskerprops={"linewidth": 2},
                medianprops={"color": "black", "linewidth": 2},
                ax=ax,
)

# Stacked labels, following the notebook style.
stacked_label_map = {
                "Best match": "Best candidate\nMCES",
                "Modified Cosine": "Modified\ncosine",
                "MS2DeepScore": "MS2DeepScore",
                "No metadata SIMBA": "No metadata\nSIMBA",
                "metadata enhanced SIMBA": "Metadata enhanced\nSIMBA",
}

ax.set_xticks(range(len(plot_labels)))
ax.set_xticklabels(
                [stacked_label_map.get(label, label) for label in plot_labels],
                fontsize=8,
)

ax.set_ylabel("Normalized MCES distance")

# Match the notebook's display range unless the original clipping switch
# explicitly requests the full normalized [0, 1] range.
if CLIP_NORMALIZED_VALUES:
                ax.set_ylim(0, 1)
else:
                ax.set_ylim(-0.05, 0.7)

# Same soft grid used in the notebook.
ax.grid(alpha=0.3)

plt.tight_layout()



# =============================================================================
# Save figure
# =============================================================================

fig.savefig(
                OUTPUT_PLOT,
                dpi=300,
                bbox_inches="tight",
)

plt.show()

print(
                f"\nSaved plot to: "
                f"{OUTPUT_PLOT}"
)



# =============================================================================
# Summary statistics
# =============================================================================

print(
                "\nSummary statistics:"
)

for method_label, values in zip(
                plot_labels,
                plot_data,
):

                print(
                                f"{method_label}: "
                                f"n={len(values)}, "
                                f"median={np.median(values):.4f}, "
                                f"mean={np.mean(values):.4f}"
                )


