# ElbowSig

## 1. Overview

`ElbowSig` implements a **generalized Elbow Significance Method** for determining the optimal number of clusters in unsupervised learning models.

It provides a unified framework to compute an **inertia-like heterogeneity measure** and assess the **statistical significance** of the “elbow” (change in slope) using permutation testing.

The implementation supports multiple popular clustering algorithms:
- **K-Means**
- **Agglomerative Clustering**
- **Fuzzy C-Means (FCM)**
- **Gaussian Mixture Models (GMM)**

Other algorithms can be implemented by the user defining an appropriate function to calculate the heterogeneity of a clustering. 

---

## 2. Features

- Computes heterogeneity (inertia-like measure) across cluster numbers \(k = 1, \dots, k_{\max}\)
- Performs **randomization-based significance testing** of the elbow
- Returns percentile thresholds and p-values
- Plots three diagnostics:
  1. Heterogeneity vs. \(k\)
  2. Slope change (Δk) vs. \(k\)
  3. p-values vs. \(k\)
- Flexible: can plug in custom heterogeneity functions

---

## 3. Dependencies

```bash
pip install numpy matplotlib scikit-learn scikit-fuzzy
```

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

Computes fuzzy clustering objective function $J_m$, analogous to inertia in hard partitions.

| Parameter | Description                           |
| --------- | ------------------------------------- |
| `pars[0]` | fuzziness exponent `m` (default=2)    |
| `pars[1]` | convergence tolerance (default=0.005) |
| `pars[2]` | max iterations (default=1000)         |

**Returns:**
`float` – $J_m$

- `GMM_Heterogeneity(X, k, random_state=42, pars=['full'])`

Computes negative log-likelihood ($-ln L$) for a Gaussian Mixture Model.

| Parameter | Description                                                  |
| --------- | ------------------------------------------------------------ |
| `pars[0]` | covariance type: `'full'`, `'tied'`, `'diag'`, `'spherical'` |

**Returns:**
float – total negative log-likelihood

--

### Heterogeneity as a function of $k$

`Heterogeneity_Total(X, kmax, random_state=42, heterogeneity_func=FCM_Heterogeneity, pars=[2, 0.005, 1000])`

Computes heterogeneity measures, slopes, and slope changes for $k=1,2,\dots,k_{\text{max}}$

**Returns:**
* hetero_k – array of heterogeneity values
* slope – first differences
* slope_change – second-order slope change
* k_values – array of k-values for slope changes

### Elbow statistic analysis

`Elbow_significance_general(X, kmax=10, nr=100, alpha=0.05, plotYN=True, random_state=42, heterogeneity_func=FCM_Heterogeneity, pars=[2, 0.005, 1000])`

Performs the full elbow significance test:
1. Computes heterogeneity curve for the original data
2. Generates uniformly distributed unstructured datasets
3. Calculates slope-change distributions under the null
4. Computes percentile thresholds and p-values

| Parameter            | Description                                     |
| -------------------- | ----------------------------------------------- |
| `kmax`               | Maximum number of clusters                      |
| `nr`                 | Number of random permutations                   |
| `alpha`              | Significance level (default 0.05)               |
| `heterogeneity_func` | Heterogeneity measure function to use           |
| `pars`               | Parameters passed to the heterogeneity function |

**Returns:**
Dictionary containing:
* 'k_values' – array of k-values
* 'hetero_k_stat' – heterogeneity per k
* 'slope_change' – slope change statistics
* 'percentile' – 1−α percentile of randomized slope changes
* 'p_values' – empirical p-values
* 'k_optimal' – selected optimal K
* 'slope_change_distribution' – all randomized slope changes
