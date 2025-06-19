import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from scipy.special import expit
import pandas as pd
import os
import matplotlib.ticker as mticker
from scipy.special import expit

# ─── File Directory Path ─────────────────────────────────

this_dir = os.path.dirname(os.path.abspath(__file__))

plots_dir = os.path.join(this_dir, "Outputs_Nested")

if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir) 

# ─── Input Data ───────────────────────────────────────

y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])        # VF events
n = np.array([1, 1, 1, 1, 1, 30, 13, 1, 103, 1])    # Number of patients per study
x = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 0])        # 0 = case report, 1 = cohort

N_studies = len(y)
n_types = len(np.unique(x))

# ─── Study Labels  ─────────────────────────────────────────────────

study_labels = [
    "Sheinman & Evans (1982)",
    "Schutzenberger (1986)",
    "Boriani (1996)",
    "Panduranga (2012)",
    "Leiria (2012)",
    "Ren (2020)",
    "Acharya (2020)",
    "Takahira (2020)",
    "Alizadeh (2022)",
    "Payami (2023)"
]

# ─── Bayesian Meta-Regression Model ────────────────────────────────────────────

with pm.Model() as model:

    # Define global mean log-odds (global intercept)
    mu = pm.Normal("mu", mu=-9.21, sigma=2.0)

    # Define between-type (accross studies) SD (heterogenity adjustment)
    sigma = pm.HalfNormal("sigma", sigma=2.0)
    
    # Define the non-centered parameterization for type-level effects
    eta_raw = pm.Normal("eta_raw", mu=0, sigma=1, shape=n_types)
    eta = pm.Deterministic("eta", eta_raw * sigma)

    # Define the Within-type (study-level) SD (heterogenity adjustment)
    tau = pm.HalfNormal("tau", sigma=2.0)
    
    # Define the non-centered parameterization for study-level effects
    zeta_raw = pm.Normal("zeta_raw", mu=0, sigma=1, shape=N_studies)
    zeta = pm.Deterministic("zeta", zeta_raw * tau)

    # Define the logit equation for each study (cohort or case)
    logit_p = mu + eta[x] + zeta
    p = pm.Deterministic("p", pm.math.sigmoid(logit_p))

    # Define the use of a Binomial Likelihood Method
    y_obs = pm.Binomial("y_obs", n=n, p=p, observed=y)

    # Perform Inference 
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        init="adapt_diag", # Change the (diagonal) mass matrix during tuning to match the variance of the posterior samples that have been examined thus far
        target_accept=0.99,
        return_inferencedata=True
    )

# Determine the uncertainity in the parameters by computing predictions of data based solely on the prior predictive distribution before we have observed any actual data
with model:
    prior_idata = pm.sample_prior_predictive(
        samples=1000, 
        return_inferencedata=True)

print(type(prior_idata))

# Determine the distribution of future observations based on the observed data and the posterior distribution of the model parameters (for later model checking against observed data)
with model:
    post_pred = pm.sample_posterior_predictive(trace, extend_inferencedata=True, random_seed=9246) # The result of sample_posterior_predictive() is attached into trace under .posterior_predictive via extend_inferencedata for later use in PPC

# ─── Computation of Paramater Values ───────────────────────────────────────────────

# Define the ratio of observed events from the raw data (count of events divided by sum of patients)
observed_rate = y.sum() / n.sum()

# Extract the posterior logit probabilities for each study
mu_samples = trace.posterior["mu"].stack(samples=("chain", "draw")).values          # Shape: (S,) - (8000,)
eta_samples = trace.posterior["eta"].stack(samples=("chain", "draw")).values        # Shape: (n_types, S) - (2, 8000) 
zeta_samples = trace.posterior["zeta"].stack(samples=("chain", "draw")).values      # Shape: (n_studies, S) - (10, 8000)

# Compute posterior VF risk for case (x=0) and cohort (x=1)
vf_case_samples   = expit(mu_samples + eta_samples[0])  # type 0 = case
vf_cohort_samples = expit(mu_samples + eta_samples[1])  # type 1 = cohort

# Extract the first column of mu_samples to define the total number of posterior samples (chains x draws) 
mu_shape = mu_samples.shape[0] # mu_shapes should equate to 8000 under current sampling methods (4 chains x 2000 draws)

# Create an empty matrix to hold the computed logit values (log-odds) for each study (with each row a study; each column a posterior sample [draw])
logits_matrix = np.zeros((N_studies, mu_shape)) 

# Compute the full logit for each study: logit(p_i) = μ + η_type[i] + ζ_i
for i in range(N_studies):
    logits_matrix[i] = mu_samples + eta_samples[x[i]] + zeta_samples[i] # eta[x[i]] defines prior_eta row (1 or 2) and zeta[i] defines prior_zeta row (range: 0-9)

# Convert the logit matrix to probabilities (This will form a matrix of posterior probabilities for VF in each study across S posterior draws)
p_study = expit(logits_matrix)

# Compute an array (weights) where each element is the proportion of patients from that study (patient count in study/total number of patients across all studies)
weights = n / n.sum()

# Compute a weighted average over the studies (axis=0) for each posterior sample (for each sample S) in Probability Scale
posterior_weighted = np.average(p_study, axis=0, weights=weights)

# Compute a weighted average over the studies (axis=0) for each posterior sample (for each sample S) in Logit Scale
posterior_weighted_log = np.log(posterior_weighted / (1 - posterior_weighted))

# Define studies that are only of the cohort-type
cohort_mask = x == 1  

# Define a new matrix in which we only contain posterior VF probabilities for cohort studies
p_study_cohort = p_study[cohort_mask] 

# Define a new matrix in which we only contain posterior VF logit values for cohort studies
l_study_cohort = logits_matrix[cohort_mask] 

#  Compute an array (weights) where each element is the proportion of patients from the cohort study (patient count in study/total number of patients across all studies)
n_cohort = n[cohort_mask] # The total number of patients in cohort studies
weights_cohort = n_cohort / n_cohort.sum()

# Compute a weighted average over the cohort studies (axis=0) for each posterior sample (for each sample S) in both logits scale and probability scale
posterior_weighted_cohort = np.average(p_study_cohort, axis=0, weights=weights_cohort)
posterior_weighted_cohort_log = np.log(posterior_weighted_cohort / (1 - posterior_weighted_cohort))

# ─── Model Criticism ───────────────────────────────────────────────

# Extract prior samples previously defined "prior_idata"
prior_mu = prior_idata.prior["mu"].stack(draws=("chain", "draw")).values        # Shape: (8000,)
prior_eta = prior_idata.prior["eta"].stack(samples=("chain", "draw")).values    # Shape: (2, 8000)
prior_zeta = prior_idata.prior["zeta"].stack(draws=("chain", "draw")).values    # Shape: (10, 8000)

# Define a matrix of shape: N (number of studies) for row; S (number of sample) for columns
prior_logits_matrix = np.zeros((N_studies, prior_mu.shape[0]))

# Compute prior VF risk on logit scale across all studies in matrix 
for i in range(N_studies):
    prior_logits_matrix[i] = prior_mu + prior_eta[x[i]] + prior_zeta[i] # eta[x[i]] defines prior_eta column (1 or 2) and zeta[i] defines prior_zeta column (range: 0-9)

# Covert the prior VF risk to probailities scale for log-scale
prior_vf_risk = expit(prior_logits_matrix) # The shape of the matrix: 

# Filter the prior VF Risk to cohort studies only
prior_p_study_cohort = prior_vf_risk[cohort_mask] 
prior_weights_cohort = n[cohort_mask] / np.sum(n[cohort_mask])

# Compute weighted average VF risk across studies per sample 
prior_weighted_cohort = np.average(prior_p_study_cohort, axis=0, weights=prior_weights_cohort)
prior_weighted_cohort_clipped = np.clip(prior_weighted_cohort, -1, 0.05)

# Define the figure size and the use of a sub-axis 
plt.figure(figsize= (10,6))

# Define kernel density estimation plot for prior distribution of VF risk (μ; logits scale)
sns.kdeplot(
    prior_weighted_cohort_clipped, 
    label="Weighted Cohort Prior (μ)", 
    fill=True, 
    color="gray", 
    alpha=0.4
    # ax=ax1,
    )

"""
# Define kernel density estimation plot for posterior distribution of VF risk across all studies (logit(p_i) = μ + η_type[i] + ζ_i; logits scale)
sns.kdeplot(
    posterior_weighted, 
    label="Weighted Posterior", 
    fill=True, 
    color="darkred", 
    alpha=0.4,
    ax=ax1
    )
"""

# Define kernel density estimation plot for posterior distribution of VF risk across cohort studies (logit(p_i) = μ + η_type[i] + ζ_i; logits scale)
sns.kdeplot(
    posterior_weighted_cohort, 
    label="Weighted Cohort Posterior", 
    fill=True, 
    color="red", 
    alpha=0.4
    # ax=ax1,
    )

# Convert the observed event rate to the log-scale for plotting
observed_rate_log = np.log(observed_rate / (1 - observed_rate))

# Define a line through graph representing the observed event rate 
plt.axvline(
    x=observed_rate, 
    color="black", 
    linestyle="--", 
    label="Observed VF Rate"
    )

# Define labels and legend for kernel density estimation plot
plt.xlabel("VF Risk")
plt.ylabel("density")
plt.legend()
plt.grid(False)
plt.tight_layout()

# Save the graph to previously defined plots_dir
plt.savefig(os.path.join(plots_dir, "prior_vs_posterior_vs_observed.png"))
plt.close()

# ─── Attempt at ECDF Plot Diagnostics ───────────────────────────────────────────────

# Define the figure size and the use of a sub-axis 
plt.figure(figsize= (10,6))

# Define cumulative distribution frequency plot for posterior distribution of VF risk across cohort studies (logit(p_i) = μ + η_type[i] + ζ_i; logits scale)
sns.ecdfplot(
    posterior_weighted_cohort, 
    label="Cohort Posterior",
    color = "green"
    )

# Define cumulative distribution frequency plot for prior distribution of VF risk across cohort studies (logit(p_i) = μ + η_type[i] + ζ_i; logits scale)
sns.ecdfplot(
    prior_weighted_cohort, 
    label="Prior", 
    color="gray"
    )

# Define a line through graph representing the observed event rate 
plt.axvline(
    observed_rate, 
    color="black", 
    linestyle="--", 
    label="Observed VF Rate")

# Define labels and legend for emperical cumulative distribution frequency plot
plt.xscale('log')
plt.xlim(1e-8, 1)
plt.xlabel("VF Risk (log scale)")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.tight_layout()


# Save the graph to previously defined plots_dir
plt.savefig(os.path.join(plots_dir, "ECF Plot.png"))
plt.close()

# ─── ArviZ Diagnostics ───────────────────────────────────────────────

# Summarize the posterior distribution of all key parameters
summary = az.summary(trace, var_names=["mu", "eta", "zeta"], round_to=4)
ess_rhat = summary[["ess_bulk", "ess_tail", "r_hat"]]
print(summary)

# Save the posterior distribution summary to a CSV at the previously defined plots_dir
posterior_csv_path = os.path.join(plots_dir, "ARVIZ Summary")  
summary.to_csv(posterior_csv_path, index=False)

# 2. Plot the posterior trace to assess model convergence and mixing
az.plot_trace(trace, var_names=["mu", "eta", "zeta"])
fig_trace = plt.gcf()

# Save the posterior trace plot to a CSV at the previously defined plots_dir
trace_path = os.path.join(plots_dir, "az_plot_trace.png")
fig_trace.savefig(trace_path)
plt.close(fig_trace)

# 5. Perform a Posterior Predictive Check on the Binomial Data
az.plot_ppc(trace, num_pp_samples=100) # Trace contains both trace.posterior_predictive["y_obs"] and trace.observed_data["y_obs"] for comparison (due to earlier extension of inference data)
fig_PPC = plt.gcf()

# Save the posterior predictive check to a CSV at the previously defined plots_dir
PPC_path = os.path.join(plots_dir, "posterior_predictive_check.png")
fig_PPC.savefig(PPC_path)
plt.close()

# ─── ArivZ Scatter Plot of Change in Prior vs Posterior Probabilitues ──────────────────────

# Define a dictionary to sort posterior distributions of μ and η
posterior_dict = {
    "mu": mu_samples,
    "eta_case": eta_samples[0],
    "eta_cohort": eta_samples[1]
}

# Convert dictionary to ArviZ InferenceData objects
posterior = az.from_dict (posterior=posterior_dict)

# Plot a scatter plot of Posterior Distribution using ArivZ InferenceData objects
az.plot_pair(
    posterior,
    var_names=["mu","eta_cohort"],
    kind='scatter',
    marginals=True,
    divergences=False,
    figsize=(6,6)
)

# Save the ArivZ scatter plot of posterior distribution
scatter_posterior = plt.gcf()
posterior_scatter_path = os.path.join(plots_dir, "az_scatter_posterior.png") 
scatter_posterior.savefig(posterior_scatter_path)
plt.close(scatter_posterior)

# ─── Seabron Scatter Plot of Change in Prior vs Posterior Probabilitues for μ vs η_cohort ──────────────────────

# Create figure size for posterior vs prior plot for μ vs η_cohort using Seaborn
plt.figure(figsize=(8, 6))

# Define variables for "prior" scatter plot
sns.scatterplot(
    x = prior_eta[1], 
    y = prior_mu, 
    alpha=0.2, 
    label='Prior', 
    color='blue'
)

# Define variables for "posterior" scatter plot
sns.scatterplot(
    x = eta_samples[1], 
    y = mu_samples, 
    alpha=0.2, 
    label='posterior', 
    color='green'
)

# Define labels and legend for joint scatter plot
plt.xlabel("η (Cohort Effect)")
plt.ylabel("μ (Global Intercept)")
plt.title("Prior vs Posterior: μ vs η (Cohort)")
plt.legend()
plt.grid(False)
plt.tight_layout()

# Save figure using previously defined plots_dir
scatter_path = os.path.join(plots_dir, "mu_vs_eta_cohort_prior_posterior.png")
plt.savefig(scatter_path)
plt.close()

# ─── Posterior Risk Summaries ─────────────────────────────────────────

# Define a summary function to compute relevant metrics for each parameter
def summarize(samples):
    return {
        "median": np.median(samples),
        "lower": np.percentile(samples, 2.5),
        "upper": np.percentile(samples, 97.5)
    }

# Compute summaries of each parameter using previously defined summary function
summary_case = summarize(vf_case_samples)
summary_cohort = summarize(vf_cohort_samples)
summary_weighted = summarize(posterior_weighted)
summary_cohort_weighted = summarize(posterior_weighted_cohort)

# Build a summary DataFrame to store all relevant values
summary_df = pd.DataFrame({
    "Group": ["Case Reports", "Cohort Studies", "Patient-Weighted Overall", "Cohort Patient-Weighted"],
    "Median VF Risk": [summary_case["median"], summary_cohort["median"], summary_weighted["median"], summary_cohort_weighted["median"]],
    "2.5%": [summary_case["lower"], summary_cohort["lower"], summary_weighted["lower"], summary_cohort_weighted["lower"]],
    "97.5%": [summary_case["upper"], summary_cohort["upper"], summary_weighted["upper"], summary_cohort_weighted["upper"]]
})

# Save summary CSV using previously defined plots_dir
summary_csv_path = os.path.join(plots_dir, "summary_data.csv")
summary_df.to_csv(summary_csv_path, index=False)

# ─── Forst Plot of Posterior Risk Estimate for Each Study ─────────────────────────────────────────

# Set up the Forest plot parameters
study_colors = ['#1f77b4' if x_i == 0 else '#ff7f0e' for x_i in x]  # blue for case studies' orange for cohort studies
means = p_study.mean(axis=1)
lower = np.percentile(p_study, 2.5, axis=1)
upper = np.percentile(p_study, 97.5, axis=1)

# Define parameters for each study within the forest plot 
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(len(means)):
    ax.errorbar(
        means[i], 
        study_labels[i],
        xerr=[[means[i] - lower[i]], [upper[i] - means[i]]],
        fmt='o', 
        color=study_colors[i], 
        ecolor=study_colors[i], 
        capsize=3
        )

# Define labels and legend for the forest plot
ax.set_xscale("log")
ax.set_xlabel("Estimated VF Risk (Posterior Mean and 95% CrI)", fontsize=12)
ax.tick_params(axis='y', labelsize=10)
ax.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

# Save figure using previously defined plots_dir
forest_plot_path = os.path.join(plots_dir, "forest_plot.png")
plt.savefig(forest_plot_path)
plt.close()
