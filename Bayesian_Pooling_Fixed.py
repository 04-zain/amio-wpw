import arviz as az
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
import pandas as pd
from typing import Dict
import os

this_dir = os.path.dirname(os.path.abspath(__file__))

plots_dir = os.path.join(this_dir, "Outputs")
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

# ─── Input Data from Review ─────────────────────────────────────────────

y = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])  # VF events
n = np.array([1, 1, 1, 1, 1, 30, 13, 1, 103, 1])  # Number of patients per study
x = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1, 0])  # Study type: 0 = case report, 1 = cohort

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
    "Payami (2023)",
]

N_studies = len(y)

# ─── Bayesian Meta-Regression Model ───────────────────────────────────

with pm.Model() as model:

    mu = pm.Normal("mu", mu=-4.6, sigma=1.16)
    beta = pm.Normal("beta", mu=0, sigma=2)
    tau = pm.HalfNormal("tau", sigma=1)

    # Non-centered reparameterization for study random effects
    u_raw = pm.Normal("u_raw", mu=0, sigma=1, shape=N_studies)
    u = pm.Deterministic("u", u_raw * tau)

    theta = mu + beta * x + u
    p = pm.Deterministic("p", pm.math.sigmoid(theta))

    y_obs = pm.Binomial("y_obs", n=n, p=p, observed=y)

    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        cores=4,
        target_accept=0.99,
        init="adapt_diag",
        return_inferencedata=True,
    )

# Sample from prior predictive to check prior implications

with model:
    prior_idata = pm.sample_prior_predictive(samples=1000, return_inferencedata=True)

print(type(prior_idata))

# Sample from posterior predictive

with model:
    post_pred = pm.sample_posterior_predictive(
        trace, extend_inferencedata=True, random_seed=92
    )

# trace.add_groups(posterior_predictive=post_pred)

# ─── Model Criticism ───────────────────────────────────────────────

# Observed data summary
observed_rate = y.sum() / n.sum()

# Extract weighted posterior (μ + β * x̄) and transform
posterior_weighted_samples = (
    expit(trace.posterior["mu"] + trace.posterior["beta"] * np.mean(x))
    .stack(samples=("chain", "draw"))
    .values
)

# Extract prior samples properly
prior_mu_samples = prior_idata.prior["mu"].stack(draws=("chain", "draw")).values
prior_beta_samples = prior_idata.prior["beta"].stack(draws=("chain", "draw")).values

# Compute prior VF risk on logit scale
x_mean = np.mean(x)
prior_logits = prior_mu_samples + prior_beta_samples * x_mean
prior_vf_risk = expit(prior_logits)

# Plotting
plt.figure(figsize=(10, 6))

sns.kdeplot(prior_mu_samples, label="Prior (μ)", fill=True, color="gray", alpha=0.4)
sns.kdeplot(
    posterior_weighted_samples,
    label="Posterior (μ + βx)",
    fill=True,
    color="red",
    alpha=0.4,
)
plt.axvline(x=observed_rate, color="black", linestyle="--", label="Observed VF Rate")
plt.title("Prior vs Posterior vs Observed VF Risk")
plt.xlabel("VF Risk")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(plots_dir, "prior_vs_posterior_vs_observed.png"))
plt.close()


# ─── ArviZ Diagnostics ───────────────────────────────────────────────

import arviz as az

# 1. Summarize key parameters
summary = az.summary(trace, var_names=["mu", "beta", "tau"], round_to=4)
ess_rhat = summary[["ess_bulk", "ess_tail", "r_hat"]]
print(summary)

posterior_csv_path = os.path.join(plots_dir, "ARVIZ Summary")
summary.to_csv(posterior_csv_path, index=False)

# Recalculate and output posterior summaries for each VF risk group
from scipy.special import expit

# 2. Plot trace to check convergence and mixing
az.plot_trace(trace, var_names=["mu", "beta", "tau"])
fig_trace = plt.gcf()

trace_path = os.path.join(plots_dir, "az_plot_trace.png")
fig_trace.savefig(trace_path)
plt.close(fig_trace)

# 3. Posterior distributions
az.plot_posterior(trace, var_names=["mu", "beta", "tau"], hdi_prob=0.95)
fig_posterior = plt.gcf()

posterior_path = os.path.join(plots_dir, "az_plot_posterior.png")
fig_posterior.savefig(posterior_path)
plt.close(fig_posterior)

# 4. Model comparison (if you fit multiple models)
# az.loo(trace)

# 5. Posterior Predictive Check (optional, if you stored ppc)

az.plot_ppc(trace, num_pp_samples=100)
plt.title("Posterior Predictive Check1")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "posterior_predictive_check1.png"))
plt.close()

az.plot_ppc(trace, group="posterior", kind="kde", figsize=(10, 6))
plt.title("Posterior Predictive Check2")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "posterior_predictive_check2.png"))
plt.close()


# ─── 4. Summary Statistics for ESS and R-hat ────────────────────────────────

ess_rhat = summary[["ess_bulk", "ess_tail", "r_hat"]]
ess_rhat.to_csv(os.path.join(plots_dir, "ess_rhat_summary.csv"))

# ─── 4. Summary Statistics of Posterior Parameters ────────────────────────────────

ppc_summary = az.summary(trace, var_names=["mu", "beta", "tau"])
ppc_summary.to_csv(os.path.join(plots_dir, "ppc_summary.csv"))

# ─── Extract Samples for Prior vs Posterior Comparison ───────────────────

# Posterior samples
mu_samples = trace.posterior["mu"].stack(draws=("chain", "draw")).values
beta_samples = trace.posterior["beta"].stack(draws=("chain", "draw")).values
u_samples = trace.posterior["u"].stack(samples=("chain", "draw")).values
logits_sample = mu_samples + beta_samples * x_mean
posterior_vf_risk = expit(logits_sample)

S = mu_samples.shape[0]
mu_mat = np.tile(mu_samples, (N_studies, 1))
beta_mat = np.tile(beta_samples, (N_studies, 1))
x_mat = np.tile(x[:, None], (1, S))

theta = mu_mat + beta_mat * x_mat + u_samples
p_post = expit(theta)
weighted_posterior = (p_post * n[:, None]).sum(axis=0) / n.sum()

p_new_patient = expit(mu_samples + beta_samples * x_mean)

cohort_mask = x == 1

p_post_cohort = p_post[cohort_mask]  # shape: (n_cohort_studies, n_draws)
n_cohort = n[cohort_mask][:, None]  # shape: (n_cohort_studies, 1)
weighted_posterior_cohort_only = (p_post_cohort * n_cohort).sum(axis=0) / n[
    cohort_mask
].sum()

# ─── Estimate Prior Distributions ──────────────────────────────────────

prior_mu = prior_idata.prior["mu"].stack(draws=("chain", "draw")).values
prior_beta = prior_idata.prior["beta"].stack(draws=("chain", "draw")).values
prior_tau = prior_idata.prior["tau"].stack(samples=("chain", "draw")).values

# ─── Data Proportions (Observed VF/n) ──────────────────────────────────

observed_case_vf = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
observed_case_n = np.array([1, 1, 1, 1, 1, 30, 13, 1, 103, 1])
case_mask = x == 0
cohort_mask = x == 1
case_prop = observed_case_vf[case_mask] / observed_case_n[case_mask]
cohort_prop = observed_case_vf[cohort_mask] / observed_case_n[cohort_mask]
# Observed VF rate
observed_rate = y.sum() / n.sum()

# ─── Posterior Density Plots ─────────────────────────────────────────

plt.figure(figsize=(12, 7))

# Plot Prior vs Posterior vs Observed VF Risk
plt.figure(figsize=(10, 6))
sns.kdeplot(prior_vf_risk, label="Prior (μ + βx̄)", fill=True, color="gray", alpha=0.4)
sns.kdeplot(
    posterior_vf_risk, label="Posterior (μ + βx̄)", fill=True, color="red", alpha=0.4
)
plt.axvline(x=observed_rate, color="black", linestyle="--", label="Observed VF Rate")
plt.title("Prior vs Posterior vs Observed VF Risk")
plt.xlabel("VF Risk")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "prior_vs_posterior_vs_observed.png"))
plt.close()

# Final Plot Settings
plt.xlabel("Estimated VF Risk")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()

KDE_plot = os.path.join(plots_dir, "KDE_plot")
plt.savefig(KDE_plot)
plt.close()


# ─── Posterior Density Plots ─────────────────────────────────────────

# Plot Prior vs Posterior vs Observed VF Risk
plt.figure(figsize=(10, 6))
sns.kdeplot(prior_vf_risk, label="Prior (μ + βx̄)", fill=True, color="gray", alpha=0.4)
sns.kdeplot(
    posterior_vf_risk, label="Posterior (μ + βx̄)", fill=True, color="red", alpha=0.4
)
plt.axvline(x=observed_rate, color="black", linestyle="--", label="Observed VF Rate")
plt.title("Prior vs Posterior vs Observed VF Risk")
plt.xlabel("VF Risk")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "prior_vs_posterior_vs_observed.png"))
plt.close()

# Final Plot Settings
plt.xlabel("Estimated VF Risk")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()

KDE_plot = os.path.join(plots_dir, "KDE_plot")
plt.savefig(KDE_plot)
plt.close()

# ─── Spread and change in Variables ─────────────────────────────────────────

n_studies = len(x)
n_draws = len(mu_samples)

print("mu_samples shape:", mu_samples.shape)
print("u_samples shape:", u_samples.shape)
print("beta_samples shape:", beta_samples.shape)

# Compute full log-odds per draw per study:
log_odds = np.zeros((n_draws, n_studies))
for i in range(n_studies):
    log_odds[:, i] = mu_samples + u_samples[i, :] + beta_samples * int(x[i])

# Convert to long format for seaborn
data = []
for i in range(n_studies):
    for d in range(n_draws):
        data.append(
            {
                "study": i,
                "logit_risk": log_odds[d, i],
                "type": "Cohort" if x[i] == 1 else "Case Report",
            }
        )
df = pd.DataFrame(data)

# Plot

plt.figure(figsize=(6, 6))

sns.violinplot(data=df, x="type", y="logit_risk")

plt.tight_layout()
violenplot_path = os.path.join(plots_dir, "interrelation_panels.png")
plt.savefig(violenplot_path)
plt.close()

# ___________________________________________________________

import arviz as az
import numpy as np

posterior_dict = {"mu": mu_samples, "beta": beta_samples}

prior_dict = {"mu": prior_mu, "beta": prior_beta}

prior = az.from_dict(prior=prior_dict)
posterior = az.from_dict(posterior=posterior_dict)

combined = az.concat(posterior, prior)

az.plot_pair(
    combined,
    var_names=["mu", "beta"],
    kind="scatter",
    marginals=True,
    divergences=False,
    coords={"chain": [0]},
    figsize=(6, 6),
)

scatter_posterior = plt.gcf()

posterior_scatter_path = os.path.join(plots_dir, "az_scatter.png")
scatter_posterior.savefig(posterior_scatter_path)
plt.close(scatter_posterior)

# Plot 1: β vs β²

beta_squared = beta_samples**2
prior_beta_squared = prior_beta**2

plt.figure(figsize=(12, 7))

# Plot 3: β vs u₁

sns.scatterplot(x=prior_beta, y=prior_mu, alpha=0.1, label="Prior", color="blue")

sns.scatterplot(
    x=beta_samples, y=mu_samples, alpha=0.1, label="Posterior", color="green"
)

# Final Plot Settings
plt.xlabel("Beta")
plt.ylabel("mu")
plt.legend()
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(plots_dir, "interrelation_panels.png")
plt.savefig(plot_path)
plt.close()

# ─── Posterior Risk Summaries ─────────────────────────────────────────


def summarize(samples: np.ndarray) -> Dict[str, float]:
    return {
        "median": np.median(samples),
        "lower": np.percentile(samples, 2.5),
        "upper": np.percentile(samples, 97.5),
    }


summary_overall = summarize(weighted_posterior)
summary_case = summarize(expit(mu_samples))
summary_cohort = summarize(expit(mu_samples + beta_samples))

# ─── Improved Forest Plot ─────────────────────────────────────────────────────

# Define color schemes
study_colors = [
    "#1f77b4" if x_i == 0 else "#ff7f0e" for x_i in x
]  # blue for case reports, orange for cohort
palette_violin = {
    "Case Reports": "#1f77b4",
    "Cohort Studies": "#ec7813",
    "Patient-Weighted": "#2ca02c",
}

fig, ax = plt.subplots(figsize=(10, 6))
means = p_post.mean(axis=1)
lower = np.percentile(p_post, 2.5, axis=1)
upper = np.percentile(p_post, 97.5, axis=1)

for i in range(len(means)):
    ax.errorbar(
        means[i],
        study_labels[i],
        xerr=[[means[i] - lower[i]], [upper[i] - means[i]]],
        fmt="o",
        color=study_colors[i],
        ecolor=study_colors[i],
        capsize=3,
    )

ax.set_xscale("log")
ax.set_xlabel("Estimated VF Risk (Posterior Mean and 95% CrI)", fontsize=12)
ax.set_title("Forest Plot of Posterior VF Risk by Study", fontsize=14)
ax.tick_params(axis="y", labelsize=10)
ax.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()

forest_plot_path = os.path.join(plots_dir, "forest_plot")
plt.savefig(forest_plot_path)
plt.close()

# Construct DataFrame of posterior samples

posterior_df = pd.DataFrame(
    {
        "VF_risk_case_reports": expit(mu_samples),
        "VF_risk_cohort_studies": expit(mu_samples + beta_samples),
        "VF_risk_patient_weighted": weighted_posterior,
    }
)

# Save to CSV for analysis or plotting in R/Python
posterior_csv_path = os.path.join(plots_dir, "posterior data")
posterior_df.to_csv(posterior_csv_path, index=False)

# Recalculate and output posterior summaries for each VF risk group
from scipy.special import expit

# Recalculate posterior samples
vf_case = expit(mu_samples)
vf_cohort = expit(mu_samples + beta_samples)
vf_weighted = weighted_posterior


# Summarize each group
def summarize(samples):
    return {
        "median": np.median(samples),
        "lower": np.percentile(samples, 2.5),
        "upper": np.percentile(samples, 97.5),
    }


summary_case = summarize(vf_case)
summary_cohort = summarize(vf_cohort)
summary_weighted = summarize(vf_weighted)
summary_cohort_weighted = summarize(weighted_posterior_cohort_only)
summary_new_patient = summarize(p_new_patient)

# Construct summary DataFrame
summary_df = pd.DataFrame(
    {
        "Group": [
            "Case Reports",
            "Cohort Studies",
            "Patient-Weighted Overall",
            "Cohort Patient-Weighted",
            "New Patient",
        ],
        "Median VF Risk": [
            summary_case["median"],
            summary_cohort["median"],
            summary_weighted["median"],
            summary_cohort_weighted["median"],
            summary_new_patient["median"],
        ],
        "2.5%": [
            summary_case["lower"],
            summary_cohort["lower"],
            summary_weighted["lower"],
            summary_cohort_weighted["lower"],
            summary_new_patient["lower"],
        ],
        "97.5%": [
            summary_case["upper"],
            summary_cohort["upper"],
            summary_weighted["upper"],
            summary_cohort_weighted["upper"],
            summary_new_patient["upper"],
        ],
    }
)

summary_csv_path = os.path.join(plots_dir, "summary data")
summary_df.to_csv(summary_csv_path, index=False)
