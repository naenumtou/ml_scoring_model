
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from catboost import CatBoostClassifier
from tqdm import TqdmExperimentalWarning

warnings.simplefilter(action = 'ignore', category = pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category = RuntimeWarning)
warnings.filterwarnings('ignore', category = UserWarning)
warnings.filterwarnings("ignore", category = TqdmExperimentalWarning)

# Hierarchical clustering
def cluster_features(
    corr_matrix: pd.DataFrame,
    distance_threshold: float = 0.3,
    method: str = 'ward'
) -> tuple[np.ndarray, pd.DataFrame]:
    
    """
    Hierarchical clustering based on mixed data correlation matrix.

    Description:
        Clustering the featurers based on distance threshold.
        Distance is 1 - |correlation|
        If threshold is 0.3 means any features having more than 0.7 correlation.
        Thoese will be grouped in the same cluster (redundant)

    Args:
        corr_matrix (pd.DataFrame)  : The mixed data correlation matrix.
        distance_threshold (float)  : Threshold for grouping features.
                                    The lower means more cluster, the higher means less cluster.
        method (str)                : The linkage algorithm to use.
                                    The method = 'ward' uses the Ward variance minimization algorithm.

    Returns:
        array           : The linkage matrix for plotting dendrogram
        pd.DataFrame    : The output of cluster analysis with cluster number mapped.

    Notes:
        - The distance threshold can be relaxed if a few features passed.
    """

    print("=== Processing ===\n[Cluster analysis]")
    
    distance_matrix = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(distance_matrix, 0)

    # Transform to condensed distance matrix
    condensed = squareform(distance_matrix, checks = False)
    linkage_matrix = linkage(condensed, method = method)
    cluster_labels = fcluster(
        linkage_matrix,
        t = distance_threshold,
        criterion = 'distance'
    )
    cluster_df = pd.DataFrame(
        {
            'feature' : corr_matrix.columns.tolist(),
            'cluster' : cluster_labels,
        }
    ).sort_values('cluster').reset_index(drop = True)

    print(f"Cluster results (Threshold: {distance_threshold * 100:.0f}%)")

    for cid, grp in cluster_df.groupby("cluster"):
        feats = grp["feature"].tolist()
        print(f"    Cluster {cid}: {feats}")

    return linkage_matrix, cluster_df

# Plot dendrogram
def plot_dendrogram(
    corr_matrix: pd.DataFrame,
    cluster_df: pd.DataFrame,
    linkage_matrix: np.ndarray,
    threshold: float
) -> None:
    
    """
    Plot dendrogram of cluster analysis.

    Description:
        Plot dendrogram of hierarchical clustering to show features containing in each cluster.

    Args:
        corr_matrix (pd.DataFrame)      : Input correlation mixed data dataframe.
        cluster_df (pd.DataFrame)       : Input cluster result.
        linkage_matrix (pd.DataFrame)   : Input the linkage matrix from cluster analysis.
        threshold (float)               : The same distance threshold used or cluster analysis.

    Returns:
        Figure: Showing figure from matplotlib.

    Notes:
        - N/A
    """

    cols = corr_matrix.columns.tolist()
    feat_cluster = cluster_df.set_index("feature")["cluster"] #Color leaf labels by cluster
    colors_cluster = plt.cm.Set2(
        np.linspace(0, 1, cluster_df["cluster"].nunique())
    )
    cluster_color_map = {
        cid: colors_cluster[i]
        for i, cid in enumerate(sorted(cluster_df["cluster"].unique()))
    }

    fig, ax = plt.subplots(figsize = (12, 6))

    # Dendrogram
    dendrogram(
        linkage_matrix,
        labels=cols,
        ax=ax,
        leaf_rotation = 90,
        leaf_font_size = 9 ,
        color_threshold=threshold,
        link_color_func = lambda _: '#888888',
    )

    for lbl in ax.get_xticklabels():
        feat = lbl.get_text()
        if feat in feat_cluster:
            cid = feat_cluster.loc[feat]
            r, g, b, _ = cluster_color_map[cid]
            lbl.set_color(
                f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            )

    # Threshold
    ax.axhline(
        y = threshold,
        color = "red",
        linestyle = "--",
        linewidth = 1
    )
    ax.text(
        0.01,
        threshold + 0.1,
        f"Threshold = {threshold * 100:.2f}%",
        transform = ax.get_yaxis_transform(),
        color = "red",
        fontsize = 8,
        verticalalignment = "bottom",
    )
    ax.set_title(
        "Hierarchical clustering dendrogram",
        fontsize = 9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    return plt.show()

# SHAP Importance
def shap_pilot_model(
    X: pd.DataFrame,
    y: pd.Series,
    features: list,
    cats_cols: list,
) -> pd.Series:
    
    """
    SHAP Importance by CatBoost.

    Description:
        SHAP Importance by the pilot model method.
        The CatBoost with non parameters tuning is leveraged as the pilot model.
        Finding the potential features by SHAP Importance.
        The result will be used as one of criteria to select based cluster result.

    Args:
        X (pd.DataFrame)    : The mixed data correlation matrix.
        y (pd.DataFrame)    : The mixed data correlation matrix.
        features (list)     : List of all selected features.
        cats_cols (list)    : List of all catercategorical features that being used to define index in dataframe.

    Returns:
        pd.Series: The output of SHAP Importance.

    Notes:
        - The CatBoost model has no parameters tuning.
        - It needs to find only potential features for the true model.
    """

    print("=== Processing ===\n[SHAP Importance]")

    X_pilot = X.copy()
    X_pilot[cats_cols] = X_pilot[cats_cols].replace([np.inf, -np.inf, np.nan], -1).astype(str)
    cat_idx = [X_pilot[features].columns.get_loc(c) for c in cats_cols]

    pilot_model = CatBoostClassifier(
        iterations = 500,
        depth = 4,
        learning_rate = 0.01,
        scale_pos_weight = (y == 0).sum() / (y == 1).sum(),
        cat_features = cat_idx,
        eval_metric = 'AUC',
        verbose = 100
    )
    pilot_model.fit(X_pilot[features], y)
    pilot_explainer = shap.TreeExplainer(pilot_model)
    shap_temp = pilot_explainer.shap_values(X_pilot[features])
    shap_importance = pd.Series(
        np.abs(shap_temp).mean(axis = 0),
        index = features,
    )

    print(f"Average SHAP Importance: {shap_importance.mean():.2f}")

    return shap_importance

# Featuers selection from cluster analysis
def select_representative(
    cluster_df: pd.DataFrame,
    shap_importance: pd.Series,
    feature_groups: dict,
    n_per_cluster: int | list = 1,
) -> list:
    
    """
    Featuers selection from cluster analysis.

    Description:
        The features selection from cluster analysis results.
        Selecting the best SHAP Importance from each cluster.
        Considering with features originated group.

    Args:
        cluster_df (pd.DataFrame)       : Input cluster result. Columns --> [feature, cluster]
        shap_importance (pd.Series)     : The shap importance from pilot model. Index --> feature, value --> shap score
        feature_groups (dict)           : Dictionary of all features groups. Dict --> {group_name: [features]}
        n_per_cluster (int, list)       : The number of features selected from each cluster.
                                        If single 'int' is applied the same for cluster.
                                        If list [2, 3, 1, ...] are applied seperated for clusters.

    Returns:
        List: The final features for the model.

    Notes:
        - N/A.
    """

    print(f"=== Result ===\nCluster selection")

    feature_to_group = {
        f: group
        for group, feats in feature_groups.items()
        for f in feats
    }
    selected = []
    cluster_ids = sorted(cluster_df['cluster'].unique())  # เรียง cluster ตามลำดับจริง
    for i, cluster_id in enumerate(cluster_ids):
        grp = cluster_df[cluster_df['cluster'] == cluster_id]
        feats = grp["feature"].tolist()

        # n ตามลำดับ index ของ cluster
        if isinstance(n_per_cluster, list):
            n = n_per_cluster[i] if i < len(n_per_cluster) else 1
        else:
            n = n_per_cluster

        if len(feats) == 1:
            selected.append(feats[0])
            print(f"    Cluster {cluster_id}")
            print(f"        [✓] Select: '{feats[0]}' (Only 1 contained)")
            continue

        ranked = (
            shap_importance
            .reindex(feats)
            .dropna()
            .sort_values(ascending=False)
            .index.tolist()
        )

        groups_in_cluster = {}
        for f in ranked:
            g = feature_to_group.get(f, 'unknown')
            groups_in_cluster.setdefault(g, []).append(f)

        chosen = []
        used = set()

        # pass 1: group representative
        for g, g_feats in groups_in_cluster.items():
            if len(chosen) >= n:
                break
            best = g_feats[0]
            chosen.append(best)
            used.add(best)

        # pass 2: shap top-up

        for f in ranked:
            if len(chosen) >= n:
                break

            if f not in used:
                chosen.append(f)
                used.add(f)

        dropped = [f for f in feats if f not in used]
        print(f"    Cluster {cluster_id}")
        print(f"        [✓] Select: {chosen}")
        print(f"            [✗] Dropped: {dropped}")
        selected.extend(chosen)

    return selected