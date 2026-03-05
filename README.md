# ElbowSig

## 1. Overview

`ElbowSig` implements an **Elbow Significance Method** for determining the optimal number of clusters in unsupervised learning models.

It provides a unified framework to compute an **inertia-like heterogeneity measure** and assess the **statistical significance** of the “elbow” (change in slope) comparing the elbow for observations with that expected for unstructured reference data (null hypothesis).

The implementation supports multiple popular clustering algorithms (option `heterogeneity_func` below):
- **K-Means**
- **Agglomerative Clustering**
- **Fuzzy C-Means (FCM)**
- **Gaussian Mixture Models (GMM)**

Other algorithms can be implemented by the user defining an appropriate function to calculate the heterogeneity of a clustering. 

Associated manuscript:

Francisco J. Perez-Reche (2026) "The elbow statistic: Multiscale clustering statistical significance", https://arxiv.org/abs/2603.03235 

---

## 2. Features

- Computes heterogeneity $H_k$ (inertia-like measure) across cluster numbers $k = 1, \dots, k_{\max}$
- Performs **randomization-based significance testing** of the elbow
- Returns percentile thresholds, p-values, and other related quantities as a function of $k$
- Plots three diagnostics:
  1. Heterogeneity vs. $k$
  2. Slope change ($\delta k$) vs. $k$
  3. Empirical p-values vs. $k$
- Flexible: can plug in custom heterogeneity functions

---

## 3. Installation

Install from source (recommended):

```bash
pip install .
```

or install dependencies directly:

```bash
pip install -r requirements.txt
```

Required runtime dependencies:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `scikit-fuzzy`
- `pandas`
- `statsmodels`

---

## 4. Functions


### 4.1. Heterogeneity for given number of clusters $k$

- `Kmeans_Heterogeneity(X, k, random_state=42, pars=[10])`

Computes total inertia (heterogeneity) for k-means clustering.

| Parameter      | Type       | Description                                    |
| -------------- | ---------- | ---------------------------------------------- |
| `X`            | array-like | Input data (n_samples × n_features)            |
| `k`            | int        | Number of clusters                             |
| `random_state` | int        | Random seed                                    |
| `pars`         | list       | `[n_init]` – number of K-Means initializations |

**Returns:**
`float` – inertia (sum of squared distances to centroids) for a k-means partition of the data in $k$ clusters

- `Agglomerative_Heterogeneity(X, k, random_state=42, pars=['euclidean', 'ward'])`

Computes total within-cluster heterogeneity for Agglomerative Clustering.

| Parameter | Description                                                      |
| --------- | ---------------------------------------------------------------- |
| `pars[0]` | distance metric (default: `'euclidean'`)                         |
| `pars[1]` | linkage method (`'ward'`, `'complete'`, `'average'`, `'single'`) |

**Returns:**
`float` – inertia (sum of squared distances to centroids) for a k-means partition of the data in $k$ clusters


- `FCM_Heterogeneity(X, k, random_state=42, pars=[2, 0.005, 1000])`

Computes fuzzy clustering objective function analogous to inertia in hard partitions.

| Parameter | Description                           |
| --------- | ------------------------------------- |
| `pars[0]` | fuzziness exponent `m` (default=2)    |
| `pars[1]` | convergence tolerance (default=0.005) |
| `pars[2]` | max iterations (default=1000)         |

**Returns:**
`float` – Inertia-like heterogeneity measure accounting for the membership of each observation to each cluster.

- `GMM_Heterogeneity(X, k, random_state=42, pars=['full'])`

Computes negative log-likelihood ($-ln L$) for a Gaussian Mixture Model.

| Parameter | Description                                                  |
| --------- | ------------------------------------------------------------ |
| `pars[0]` | covariance type: `'full'`, `'tied'`, `'diag'`, `'spherical'` |

**Returns:**
float – negative log-likelihood

--

### Heterogeneity as a function of $k$

`Heterogeneity_Total(X, kmax, random_state=42, heterogeneity_func=FCM_Heterogeneity, pars=[2, 0.005, 1000])`

Computes heterogeneity measures, slopes, and slope changes for $k=1,2,\dots,k_{\text{max}}$

**Returns:**
* hetero_k – array of heterogeneity values
* slope – first differences
* slope_change – second-order slope change
* k_values – array of k-values for slope changes


### Elbow Statistic Analysis

` Elbow_significance_general(X, kmax=10, nr=100, qperk=0.05, qFDR=0.05, plotYN=True, random_state=42, heterogeneity_func=FCM_Heterogeneity, pars=[2, 0.005, 1000], nsig=20, fsel=0.5, random_data_func=data_random)`

Performs the full elbow significance test to determine the statistically significant number(s) of clusters ($K$).

1.  Computes the heterogeneity curve for the original data.
2.  Generates uniformly distributed unstructured datasets (`nr` times).
3.  Calculates slope-change distributions ($\delta_k$ values) under the null hypothesis.
4.  Calculates percentile thresholds and empirical p-values.
5.  Computes an adaptive significance threshold ($p_{\text{sig}}$) based on $q_{\text{sig}}$ random selection of a fraction `fsel` of the `nr` reference datasets.
6.  Controls global false discovery rate (FDR) associated with test at different $k$ using Benjamini-Hochberg method.
8.  Identifies all $K$ values whose $\delta_k$ is statistically significant.

| Parameter | Description |
| :--- | :--- |
| `kmax` (default: 10) | Maximum number of clusters to test. |
| `nr` (default: 100) | Number of random permutations used to build the null distribution. |
| `qperk` (default: 0.05) | The significance level ($q_1$) for the per-scale threshold $p_{\text{sig}}$. |
| `qFDR` (default: 0.05) | The target level ($q_2$) for False Discovery Rate (FDR) control using Benjamini-Hochberg (BH). |
| `plotYN` (default: `True`) | Boolean flag to control plotting of results (default `True`). |
| `heterogeneity_func` | Heterogeneity measure function to use (e.g., `Kmeans_Heterogeneity`). Available modes: `Agglomerative_Heterogeneity`, `Kmeans_Heterogeneity`, `FCM_Heterogeneity`, `GMM_Heterogeneity`  |
| `pars` | Parameters passed to the heterogeneity function. This depends on the clustering method given in `heterogeneity_func`. Examples: `['euclidean', 'ward']` for `Agglomerative_Heterogeneity`. `[10]`, number of initializations for `Kmeans_Heterogeneity`.  `[2, 0.005, 1000]`, fuzziness, error tolerance, max iterations for `Elbow_significance_general`. `['full']`, covariance type for `GMM_Heterogeneity`.  |
| `nsig` | Parameter for the adaptive $p$-value threshold calculation (number of runs used to determine the threshold $p_{\text{sig}}$). |
| `fsel` | Fraction of realizations to sample when calculating the adaptive threshold ($p_{\text{sig}}$). |
| `random_data_func` | The function used to generate the unstructured null hypothesis data (e.g., data_random_pca_aligned for a PCA-aligned null). |

**Returns:**
A dictionary containing:

| Key | Description |
| :--- | :--- |
| `'k_values'` | Array of tested $K$ values. |
| `'hetero_k_stat'` | Heterogeneity per $K$ for the original data. |
| `'slope_change'` | Slope change statistics ($\delta_k$) for the original data. |
| `'percentile'` | $100 \times (1 - p_{\text{sig}})$ percentile of randomized slope changes. |
| `'p_values'` | Empirical p-values for each $K$. |
| `'p_values_corrected'` | $p$-values corrected for multiple comparisons (all tested $k$ values) using the Benjamini-Hochberg procedure. |
| `'k_optimal'` | List of all statistically significant $K$ values. Returns `[1]` if no significant clusters are found. |
| `'slope_change_distribution'` | All randomized slope changes (matrix of size `nr` x `kmax-1`). |
| `'pv_thres_list'` | List of $p$-value quantiles found across the $q_{\text{sig}}$ repetitions. |
| `'p_sig_thres'` | The final, minimum adaptive p-value threshold ($p_{\text{sig}}$) calculated from the `pv_thres_list`. |


### `plot_Elbow_significance_general_results(results)`

Generates a three-panel plot visualization of the Elbow Significance Test results, allowing for a comprehensive assessment of the optimal number of clusters ($K$).

**Input:**
* `results` (dict): The output dictionary returned by the `Elbow_significance_general` function.

**Output:**
* A three-panel figure displayed to the screen.

**Visualization Panels:**

| Panel | Content | Description |
| :---: | :--- | :--- |
| **Top** | **Total Heterogeneity vs. $k$** | Plots the raw "inertia-like" curve (Total heterogeneity) for the data against the number of clusters ($K$). |
| **Middle** | **Elbow statistic $\delta_k$ vs. $k$** | Compares the $\delta_k$ statistic for the real data (`Data`) against the $100(1-p_{\text{sig}})$ **Percentile** curve derived from the randomized null data. Significant $K$ values are where the `Data` curve is above the `Random` curve. |
| **Bottom** | **P-Value vs. $k$** | Plots the empirical p-values for each $K$ against the constant **adaptive threshold** $p_{\text{sig}}$ (calculated via the bootstrapped permutation test). Significant $K$ values are where the p-value is **below** the dashed red line $p_{\text{sig}}$. |

### `compare_slope_changes_scatter_results(results)`

Generates a scatter plot comparing the **data's slope change statistic** ($\delta_k$) against the full distribution of $\delta_k$ values obtained from the **randomized null data** (reference data).

This plot visually demonstrates the basis for the significance test by showing how the data's slope change compares to the expected range under the null hypothesis.

**Input:**
* `results` (dict): The output dictionary returned by the `Elbow_significance_general` function.

**Output:**
* A single figure displayed to the screen.

**Plot Elements:**

| Element | Appearance | Description |
| :--- | :--- | :--- |
| **Randomized Null Data** | Gray, semi-transparent scatter points | Represents the $\delta_k$ values for *each* of the `nr` random reference datasets, forming the **null distribution** for every tested $K$. |
| **Observed $\delta_k$ statistic** | Blue line with circles | Shows the actual $\delta_k$ values derived from the original dataset. Points lying significantly above the gray cloud indicate strong evidence of structure. |
| **Reference $\delta_k$ statistic threshold** | Orange line with squares | Plots the $100 \times (1-p_{\text{sig}})$ Percentile curve. This is the critical threshold: if the blue line (Data) is above the orange line, the corresponding $K$ is considered statistically significant. |


### `visualize_agglomerative_clustering(data, kcluster)`

Performs and plots the results of **Agglomerative Clustering** on a dataset. This function is designed to visualize the cluster assignments based on the first two dimensions of the input data, typically used for low-dimensional data or the first two Principal Components (PCA).

**Input:**

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `data` | `np.ndarray` | The dataset to cluster. The plot uses `data[:, 0]` and `data[:, 1]`. |
| `kcluster` | `int` | The target number of clusters ($K$). |

**Process:**

1.  Initializes and runs `sklearn.cluster.AgglomerativeClustering` with `n_clusters = kcluster`.
2.  Assigns cluster labels to each data point.
3.  Generates a dedicated `ListedColormap` with distinct colors for up to 30 clusters to ensure visual clarity.
4.  Creates a scatter plot using the first two columns of `data` ($x_1$ and $x_2$), colored by the assigned cluster label.
5.  Includes a colorbar to serve as the cluster legend.

**Output:**

* A Matplotlib figure displayed to the screen showing the data points colored by cluster.
* **Returns:** A tuple containing: `(labels, clustering_object)`.

### `compute_gap_statistic(data0, max_k, n_refs=10, cluster_method="KMeans", random_data_func=data_random, plot=True)`

Computes the Gap Statistic and its associated standard deviation ($s_k$) for a range of cluster numbers ($K=1$ to $K_{\text{max}}$). The function also determines the optimal $K$ using two common criteria and optionally plots the results.
**Input Parameters:**

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `data0` | `np.ndarray` | N/A | The input dataset to be clustered. |
| `max_k` | `int` | N/A | The maximum number of clusters to evaluate. |
| `n_refs` | `int` | `10` | The number of reference (null) datasets to generate for the comparison. |
| `cluster_method` | `str` | `"KMeans"` | The clustering algorithm to use. Must be `"KMeans"` or `"Agglomerate"`. |
| `random_data_func` | The function used to generate the unstructured null hypothesis data (e.g., data_random_pca_aligned for a PCA-aligned null). |
| `plot` | `bool` | `True` | If `True`, generates and displays a plot of the Gap Statistic with error bars and optimal $K$ markers. |

**Core Process:**

1.  **Standardization:** The input data is standardized using `StandardScaler` to ensure all features contribute equally to the distance metrics. Reference data is generated on the original scale and *then* standardized using the same fit.
2.  **WCSS Calculation:** For each $K$ from 1 to `max_k`, the function computes the Within-Cluster Sum of Squares (WCSS) for both the real data ($\text{WCSS}_k$) and the `n_refs` reference datasets ($\text{WCSS}_{r, k}$). The log of WCSS ($\log(\text{WCSS})$) is used for stability.
3.  **Gap Calculation:** The Gap Statistic ($\text{Gap}_k$) is computed as the difference between the average $\log(\text{WCSS})$ of the reference data ($\mathbb{E}[\log(\text{WCSS}_{r,k})]$) and the $\log(\text{WCSS})$ of the real data ($\log(\text{WCSS}_k)$).
4.  **Standard Deviation ($s_k$):** The standard deviation of the reference $\log(\text{WCSS})$ values is computed and scaled to produce $s_k$, which accounts for the variability of the null distribution.
5.  **Optimal $K$ Determination:**
    * **Original Criterion (Tibshirani et al (2001):** $K$ is the smallest value such that $\text{Gap}_k \geq \text{Gap}_{k+1} - s_{k+1}$.
    * **Maximum Gap Criterion:** $K$ is the value that maximizes $\text{Gap}_k$.

**Returns:**

| Value | Type | Description |
| :--- | :--- | :--- |
| `wcss_values` | `np.ndarray` | The array of $\log(\text{WCSS}_k)$ values for $K=1$ to $K_{\text{max}}$. |
| `gap_values` | `np.ndarray` | The array of $\text{Gap}_k$ values for $K=1$ to $K_{\text{max}}$. |
| `sk_values` | `list` | The standard deviation ($s_k$) for each $K$. |
| `optimal_k_tibshirani` | `int` | The optimal $K$ based on the conservative Tibshirani criterion. |
| `optimal_k_max_gap` | `int` | The optimal $K$ based on the maximum $\text{Gap}_k$ value. |

**`if plot=True`**

The function generates two plots to fully illustrate the Gap Statistic analysis.

1. WCSS and Expected WCSS Plot. This plot shows the raw components used to calculate the Gap Statistic: The logarithm of the Within-Cluster Sum of Squares for the real dataset, illustrating the "Elbow" curve vs. the expected log-WCSS under the null hypothesis (average log-WCSS of the random reference datasets).

2. Gap Statistic Plot. This plot shows the final metric with the criteria used for selection.


