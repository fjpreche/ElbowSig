import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import random
import math
import pandas as pd

import os

os.environ["OMP_NUM_THREADS"] = "2"
import statsmodels.sandbox.stats.multicomp as smm

# K-Means Heterogeneity
# pars: [n_init]
# elbow_measure: ['inertia']
from sklearn.cluster import KMeans
def Kmeans_Heterogeneity(X, k, random_state=42, pars=[10]):
    X = np.asarray(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=pars[0])
    kmeans.fit(X_scaled)
    return kmeans.inertia_

# Agglomerative Clustering Heterogeneity
# pars: [metric, linkage] 
#   metric - str or callable, default=”euclidean”
#   linkage{‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
from sklearn.cluster import AgglomerativeClustering
def Agglomerative_Heterogeneity(X, k, random_state=42, pars=['euclidean', 'ward']):
    X = np.asarray(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    agglo = AgglomerativeClustering(n_clusters=k,metric=pars[0],linkage=pars[1])
    labels = agglo.fit_predict(X_scaled)
    heterogeneity = 0
    for cluster in range(k):
        cluster_points = X_scaled[labels == cluster]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            heterogeneity += np.sum((cluster_points - centroid) ** 2)
    return heterogeneity


# Fuzzy C-Means Heterogeneity
# pars: [m, error, maxiter]
import skfuzzy as fuzz
def FCM_Heterogeneity(X, k, random_state=42,pars=[2, 0.005, 1000]):
    X = np.asarray(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_T = X_scaled.T
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        X_T, c=k, m=pars[0], error=pars[1], maxiter=pars[2], init=None, seed=random_state
    )
    # jm is the objective function J_m, analogous to inertia
    return jm[-1]  # final iteration's objective value

# Gaussian Mixture Model Heterogeneity
# pars: [covariance_type]  (covariance_type{‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’)
from sklearn.mixture import GaussianMixture
def GMM_Heterogeneity(X, k, random_state=42, pars=['full']):
    X = np.asarray(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=k, random_state=random_state, covariance_type=pars[0])
    gmm.fit(X_scaled)
    
    nll = -gmm.score(X_scaled) * len(X_scaled)  # total negative log-likelihood
    return nll  # negative log-likelihood


def Heterogeneity_vs_k(X, kmax, random_state=42, heterogeneity_func=FCM_Heterogeneity,pars=[2, 0.005, 1000]):
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    hetero_k = []
    for k in range(1, kmax + 1):
        val = heterogeneity_func(X_scaled, k, random_state=random_state,pars=pars)
        hetero_k.append(val)
    
    hetero_k = np.array(hetero_k)
    
    slope = [hetero_k[k]-hetero_k[k-1] for k in range(1,len(hetero_k))]
    k_values = np.array([k for k in range(2,len(slope)+1)])
    slope_change = np.array([slope[k]/slope[k+1]-1 for k in range(len(slope)-1)])

    return hetero_k, slope, slope_change, k_values

def data_random(data):
    n_samples, n_features = data.shape
    feature_min = np.min(data, axis=0)
    feature_max = np.max(data, axis=0)
    reference_data = np.random.uniform(feature_min, feature_max, size=(n_samples, n_features))
    return reference_data


def data_random_pca_aligned(data):
    """
    Generates reference data from a uniform distribution over a box aligned 
    with the principal components of the original data (Tibshirani et al., 2001).
    """
    # FIX 1: Ensure all subsequent operations use NumPy arrays
    data = np.asarray(data) 

    n_samples, n_features = data.shape

    # 1. Center the data
    data_mean = np.mean(data, axis=0) 
    data_centered = data - data_mean

    # 2. Compute SVD
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    V = Vt.T

    # 3. Rotate the centered data into the PC space
    data_rotated = data_centered @ V

    # 4. Determine the ranges (bounding box) in the PC space
    feature_min = np.min(data_rotated, axis=0)
    feature_max = np.max(data_rotated, axis=0)
    
    # -----------------------------------------------------
    # FIX 2: Reshape bounds to ensure correct broadcasting in np.random.uniform.
    # We explicitly convert (p,) vectors to (1, p) row vectors.
    feature_min = feature_min[np.newaxis, :]
    feature_max = feature_max[np.newaxis, :]
    # -----------------------------------------------------

    # 5. Generate uniform random data Z' over the bounding box in PC space
    # The 'size' parameter is correctly n x p, and low/high are now 1 x p,
    # ensuring np.random.uniform broadcasts the bounds correctly to fill the n x p array.
    reference_data_rotated = np.random.uniform(
        low=feature_min, 
        high=feature_max, 
        size=(n_samples, n_features)
    )

    # 6. Rotate Z' back to the original space
    reference_data = reference_data_rotated @ V.T

    # 7. Un-center the resulting reference data
    reference_data_uncentered = reference_data + data_mean
    
    return reference_data_uncentered

def p_thres(deltar_matrix, fsel, alpha=0.05):


    kmaxr = deltar_matrix.shape[1]
    nsel = math.ceil(fsel*len(deltar_matrix))

    selected_r = random.sample(range(len(deltar_matrix)), nsel)

    pval_r = np.zeros([len(selected_r),kmaxr])

    iselected = 0
    for r in selected_r:
        rprime = [i for i in range(len(deltar_matrix)) if i != r]
        for rp in rprime:
            for k in range(kmaxr):
                if deltar_matrix[rp,k] > deltar_matrix[r,k]:
                    pval_r[iselected,k] = pval_r[iselected,k] + 1/(len(deltar_matrix)-1)

        iselected = iselected + 1

    return pd.DataFrame(pval_r).quantile(alpha)


def min_p_thres_q_runs(deltar_matrix, fsel, nsig, alpha=0.05):
    """
    Runs the p_thres_single_run function q times and returns the minimum 
    p-value quantile for each k column across all runs.

    Args:
        deltar_matrix (pd.DataFrame): DataFrame of delta_k values (rows=realizations, columns=k).
        fsel (float): Fraction of rows to select (nsel/N).
        q (int): The number of repetitions (stochastic runs).
        alpha (float): The quantile level (e.g., 0.05).

    Returns:
        pd.Series: A Series containing the minimum alpha-quantile found for 
                   each k column across the q runs.
    """
    
    # List to hold the resulting Series (quantiles) from each run
    quantile_series_list = []
    
    #print(f"Starting {nsig} repetitions of p_thres calculation...")
    
    for run in range(nsig):
        # 1. Call the single-run function
        # This returns a pandas Series (list1)
        quantile_result = p_thres(deltar_matrix, fsel, alpha)
        #print(quantile_result)
        
        # 2. Store the result
        quantile_series_list.append(list(quantile_result))
        
    #print("Repetitions complete.")
    
    # 3. Combine all resulting Series into a single DataFrame
    # Rows are the runs, columns are the k values.
    df_quantiles = pd.DataFrame(quantile_series_list)
    
    # 4. Find the minimum value for each column (i.e., for each k)
    min_quantiles = df_quantiles.min()
    
    # 5. Return the resulting minimum Series
    return min_quantiles


def Elbow_significance_general(X, kmax=10, nr=100, qperk=0.05, qFDR=0.05, plotYN=True, random_state=42, 
                               heterogeneity_func=FCM_Heterogeneity, pars=[2, 0.005, 1000],
                               nsig=20, fsel=0.5, random_data_func=data_random):
    
    np.random.seed(random_state)
    
    # --- Original data statistic ---
    hetero_k_0, slope0, slope_change0, k_values0 = Heterogeneity_vs_k(
        X, kmax, random_state, heterogeneity_func=heterogeneity_func, pars=pars
    )

    # --- Randomized distributions (permutation test) ---
    slope_change_distribution = np.zeros((nr, len(slope_change0)))
    
    for r in range(nr):
        # *** ADAPTED LINE ***
        # Use the function passed in the random_data_func parameter
        X_rand = random_data_func(X) 
        
        # Ensure that X_rand is not passed directly to clustering functions 
        # if they expect a specific data structure or scaling.
        # Assuming Heterogeneity_vs_k handles scaling internally, which is standard.
        _, _, slope_change_r, _ = Heterogeneity_vs_k(
            X_rand, kmax, random_state + r, 
            heterogeneity_func=heterogeneity_func, pars=pars
        )
        slope_change_distribution[r, :] = slope_change_r

    # --- Adaptive level calculation ---
    pv_thres_list = min_p_thres_q_runs(slope_change_distribution, fsel, nsig, qperk)
    p_sig_thres = np.min(pv_thres_list)

    # --- Compute statistics ---
    # The percentile calculation now uses the adaptive threshold p_sig_thres
    percentile = np.percentile(slope_change_distribution, 100 * (1 - p_sig_thres), axis=0)
    p_values = np.mean(slope_change_distribution >= slope_change0, axis=0)

    # --- List of optimal k ---
    k_opt = k_values0[slope_change0 > percentile].tolist() 
    k_opt = k_opt if k_opt else [1]

    rejected, p_corrected, _, _ = smm.multipletests(
        p_values,#[0:15], 
        alpha=qFDR, 
        method='fdr_bh'
    )


    out = {
        'k_values': k_values0,
        'hetero_k_stat': hetero_k_0,
        'slope_change': slope_change0,
        'percentile': percentile,
        'p_values': p_values,
        'p_values_corrected': p_corrected,
        'rejected_H0_FDR': rejected,
        'k_optimal': k_opt,
        'slope_change_distribution': slope_change_distribution,
        'pv_thres_list': pv_thres_list,
        'p_sig_thres': p_sig_thres
    }

    if plotYN:
        # Assuming plot_Elbow_significance_general_results is defined
        plot_Elbow_significance_general_results(out)

    return out


#### -----------------------------------------------------------------
#### Additional plotting function to compare slope changes scatter
#### -----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def plot_Elbow_significance_general_results(results):
    # --- Configuration ---
    # FONT_SIZE controls the size of tick labels, legend, and title
    FONT_SIZE = 14 
    # LABEL_FONT_SIZE controls the size of the axes labels (x and y labels)
    LABEL_FONT_SIZE = 16 

    # --- Data Extraction (Assumed from 'results' dictionary) ---
    hetero_k_0 = results['hetero_k_stat']
    slope_change0 = results['slope_change']
    k_values0 = results['k_values']
    percentile = results['percentile']
    p_values = results['p_values']
    p_sig_thres = results['p_sig_thres']

    # Update rcParams for general font size (this affects title/legend)
    plt.rcParams.update({'font.size': FONT_SIZE}) 
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 9))

    # Apply tick label size to all subplots
    for axi in ax:
        # Increase the font size for both x and y tick labels
        axi.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    # --- Plot 0: "Inertia-like" curve ---
    ax[0].plot(np.arange(1, len(hetero_k_0) + 1), hetero_k_0, '-o')
    ylabel = 'Total heterogeneity'
    
    # Use fontsize argument for axes labels
    ax[0].set_xlabel('$k$', fontsize=LABEL_FONT_SIZE)
    ax[0].set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    #ax[0].set_title(f'{ylabel} vs. $k$')


    # --- Plot 1: Slope changes and percentiles ---
    ax[1].plot(k_values0, slope_change0, '-o', label='Data')
    ax[1].plot(k_values0, percentile, '-s', label=f'Unstructured reference ({100 * (1 - p_sig_thres):.0f}th Percentile)')
    
    # Use raw string r'' for robust LaTeX rendering and fontsize argument
    ax[1].set_xlabel('$k$', fontsize=LABEL_FONT_SIZE)
    ax[1].set_ylabel(r'$\delta_k$', fontsize=LABEL_FONT_SIZE) # Fixed LaTeX and increased size
    ax[1].legend()
    #ax[1].set_title('Slope Change vs. k')

    # --- Plot 2: P-values vs. k ---
    ax[2].plot(k_values0, p_values, '-o', color='black', label='p-value')
    
    # Use f-string with correct LaTeX subscript formatting
    ax[2].axhline(y=p_sig_thres, color='r', linestyle='--', 
                  label=f'$p_{{sig}}$ = {p_sig_thres:.3f}')
    
    # Use fontsize argument for axes labels
    ax[2].set_xlabel('$k$', fontsize=LABEL_FONT_SIZE)
    ax[2].set_ylabel('p-value', fontsize=LABEL_FONT_SIZE)
    ax[2].legend()
    #ax[2].set_title('P-Value vs. $k$')

    plt.tight_layout()
    plt.show()
    

def compare_slope_changes_scatter_results(results):
    """
    Creates a plot comparing the data's slope change (delta_k) against 
    individual observations and the percentile threshold derived from 
    the randomized null distribution.

    Args:
        results (dict): The output dictionary from Elbow_significance_general.
    """
    
    # --- Data Extraction ---
    k_values0 = results['k_values']
    slope_change0 = results['slope_change']
    percentile = results['percentile']
    slope_change_distribution = results['slope_change_distribution']
    p_sig_thres = results['p_sig_thres'] # Use the calculated adaptive threshold

    # --- Configuration (Matching previous requests) ---
    FONT_SIZE = 14
    LABEL_FONT_SIZE = 16
    plt.rcParams.update({'font.size': FONT_SIZE}) 
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Apply tick label size
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    # 1. Plot the individual observations of the distribution as scatter points
    for i in range(slope_change_distribution.shape[0]):
        if i == 0:
            ax.scatter(k_values0, slope_change_distribution[i, :], 
                       color='gray', alpha=0.3, s=20, 
                       label=r'Randomized Null Data ($\delta_k$ distribution)')
        else:
            ax.scatter(k_values0, slope_change_distribution[i, :], 
                       color='gray', alpha=0.3, s=20)

    # 2. Plot the original slope change values (Data)
    ax.plot(k_values0, slope_change0, 'o-',  
            linewidth=2, label='Data')

    # 3. Plot the percentile threshold
    # Note: Using p_sig_thres for the label for consistency with the adaptive method
    ax.plot(k_values0, percentile, '-s', linewidth=2,
            label=f'Unstructured reference ({100 * (1 - p_sig_thres):.1f}th Percentile)')

    # --- Formatting ---
    ax.set_xlabel('$k$', fontsize=LABEL_FONT_SIZE)
    # Use raw string and correct font size for ylabel
    ax.set_ylabel(r'$\delta_k$', fontsize=LABEL_FONT_SIZE) 
    ax.set_title('Comparison of Data Slope Change vs. Null Distribution', fontsize=LABEL_FONT_SIZE)
    ax.legend(loc='best', markerscale=1.5)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


## -----------------------------------------------------------------
## Additional plotting function to visualize Agglomerative Clustering results
## -----------------------------------------------------------------
def visualize_agglomerative_clustering(data, kcluster):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    #dataXPCA = dataX.copy()  # Use PCA-reduced data if needed, or keep original dataX

    # Agglomerative clustering on PCA-reduced data
    clustering = AgglomerativeClustering(n_clusters=kcluster)
    labels = clustering.fit_predict(data)

    # Get a continuous colormap
    cmap = plt.colormaps.get_cmap('viridis')

    # Sample 30 distinct colors from it
    all_30_colors = cmap(np.linspace(0, 1, 30))
    new_cmap = ListedColormap(all_30_colors)

    # Plot first two PCA components with cluster coloring
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=new_cmap, s=40)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'Agglomerative Clustering (k={kcluster})')

###################
## Gap statistic ##
###################

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

def compute_wcss(data, labels, centroids):
    """
    Compute the within-cluster sum of squares (WCSS).
    """
    wcss = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:  # Avoid errors for empty clusters
            wcss += np.sum((cluster_points - centroid) ** 2)
    return wcss

def generate_reference_data(data, n_refs=10, random_data_func=data_random):
    """
    Generate feature-wise reference (random) datasets.
    """

    reference_datasets = []
    for _ in range(n_refs):
        reference_data = random_data_func(data)
        reference_datasets.append(reference_data)
    return reference_datasets


import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler



def compute_gap_statistic(data0, max_k, n_refs=10, cluster_method="KMeans", random_data_func=data_random, plot=True):
    """
    Compute the gap statistic for a range of cluster numbers.
    """
    wcss_values = []
    gap_values = []
    sk_values = []  # To store standard deviations for gap statistics
    
    # Standardize the data
    scaler = StandardScaler().fit(data0)
    data = scaler.transform(data0)
    
    # Pre-generate reference datasets before standardization
    reference_datasets = generate_reference_data(data0, n_refs, random_data_func)
    reference_datasets = [scaler.transform(ref_data) for ref_data in reference_datasets]

    for k in range(1, max_k + 1):
        # Fit clustering algorithm
        if cluster_method == "KMeans":
            model = KMeans(n_clusters=k, n_init=10, random_state=42)
        elif cluster_method == "Agglomerate":
            model = AgglomerativeClustering(n_clusters=k)
        else:
            raise ValueError("Invalid cluster method. Use 'KMeans' or 'Agglomerate'.")
        
        labels = model.fit_predict(data)
        
        # Compute centroids if using KMeans
        if cluster_method == "KMeans":
            centroids = model.cluster_centers_
        else:
            centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Compute WCSS for data
        wcss = compute_wcss(data, labels, centroids)
        log_wcss = np.log(wcss) if wcss > 0 else -np.inf
        wcss_values.append(log_wcss)
        
        # Compute WCSS for reference datasets
        reference_wcss = []
        for ref_data in reference_datasets:
            ref_labels = model.fit_predict(ref_data)
            if cluster_method == "KMeans":
                ref_centroids = model.cluster_centers_
            else:
                ref_centroids = np.array([ref_data[ref_labels == i].mean(axis=0) for i in range(k)])
            wcss_ref = compute_wcss(ref_data, ref_labels, ref_centroids)
            reference_wcss.append(np.log(wcss_ref) if wcss_ref > 0 else -np.inf)
        
        # Compute gap statistic
        mean_ref_wcss = np.mean(reference_wcss)
        gap = mean_ref_wcss - log_wcss
        sk = np.sqrt(np.mean((reference_wcss - mean_ref_wcss) ** 2)) * np.sqrt(1 + 1 / n_refs)
        
        gap_values.append(gap)
        sk_values.append(sk)
    
    # Compute optimal k according to Tibshirani 2001
    optimal_k_tibshirani = 1  # Start with k=1 as a base
    for k in range(len(gap_values) - 1):
        if gap_values[k] >= (gap_values[k + 1] - sk_values[k + 1]):
            optimal_k_tibshirani = k + 1
            break
            
    # Compute optimal k based on the maximum of the Gap curve
    # Find the index of the maximum gap value, then add 1 to get the k value
    optimal_k_max_gap = np.argmax(gap_values) + 1

    if plot:
        # Example values (replace these with your computed values)
        # gap_values = np.array([...])  # Replace with actual gap statistic values
        # sk_values = np.array([...])  # Replace with actual standard errors
        # optimal_k = ...  # Replace with the computed optimal k

        # --- Configuration ---
        FONT_SIZE = 14  # Font size for tick labels, legend, and titles
        LABEL_FONT_SIZE = 16 # Larger font size for axes labels
        plt.rcParams.update({'font.size': FONT_SIZE})

        # Define range for k
        k_values = np.arange(1, len(np.array(gap_values)) + 1)

        # Plotting

        plt.figure(figsize=(8, 4))
        plt.plot(k_values, np.array(wcss_values), '-o', label='ln $ W_k$')
        plt.plot(k_values, np.array(gap_values)+np.array(wcss_values), '-s', label='$E_n($ln $ W_k^r)$')
        #plt.title('Gap Statistic vs Number of Clusters')
        plt.xlabel('$k$', fontsize=LABEL_FONT_SIZE)
        plt.ylabel('ln $ W_k$ and $E_n($ln $ W_k^r)$', fontsize=LABEL_FONT_SIZE)
        plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.errorbar(k_values, np.array(gap_values), yerr=sk_values, fmt='-o', capsize=5, label='Gap Statistic')
        plt.axvline(x=optimal_k_tibshirani, color='green', linestyle='--', label=f'Optimal k (Original)= {optimal_k_tibshirani}')
        plt.axvline(x=optimal_k_max_gap, color='red', linestyle='--', label=f'Optimal k (max)= {optimal_k_max_gap}')
        plt.title('Gap Statistic vs Number of Clusters', fontsize=LABEL_FONT_SIZE)
        plt.xlabel('$k$', fontsize=LABEL_FONT_SIZE)
        plt.ylabel('Gap Statistic')
        plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return np.array(wcss_values), np.array(gap_values), sk_values, optimal_k_tibshirani, optimal_k_max_gap
