import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Generate Synthetic Data
np.random.seed(42)  # For reproducibility

# Parameters
n = 100  # Total number of data points
changepoint_true = 60  # True changepoint index
mean1 = 0
mean2 = 5
std = 1

# Generate data
data = np.concatenate([
    np.random.normal(mean1, std, changepoint_true),
    np.random.normal(mean2, std, n - changepoint_true)
])

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(data, label='Data')
plt.axvline(changepoint_true, color='red', linestyle='--', label='True Changepoint')
plt.title('Synthetic Data with a Changepoint')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# 2. Bayesian Changepoint Detection

# Define prior: assume uniform prior for changepoint
prior = np.ones(n) / n

# Initialize arrays to store posterior probabilities
posterior = np.zeros(n)

# Define likelihoods
# Assume known standard deviation, and model data as Gaussian
for t in range(n):
    if t == 0:
        # If changepoint is at the first point, all data is from mean2
        mean_before = mean2
        mean_after = mean2
    else:
        # Calculate the mean before changepoint t
        mean_before = np.mean(data[:t])
        mean_after = np.mean(data[t:])
    
    # Compute likelihood: product of probabilities
    # To prevent underflow, use log-likelihood
    try:
        log_likelihood_before = np.sum(norm.logpdf(data[:t], mean1, std))
        log_likelihood_after = np.sum(norm.logpdf(data[t:], mean2, std))
    except:
        log_likelihood_before = -np.inf
        log_likelihood_after = -np.inf
    
    log_likelihood = log_likelihood_before + log_likelihood_after
    posterior[t] = log_likelihood + np.log(prior[t])

# Convert log-posterior to regular posterior
max_log_posterior = np.max(posterior)
posterior = np.exp(posterior - max_log_posterior)  # for numerical stability
posterior /= np.sum(posterior)

# 3. Identify the Most Probable Changepoint
changepoint_estimated = np.argmax(posterior)

# 4. Plot the Posterior Probabilities
plt.figure(figsize=(12, 6))
plt.plot(posterior, label='Posterior Probability')
plt.axvline(changepoint_true, color='red', linestyle='--', label='True Changepoint')
plt.axvline(changepoint_estimated, color='green', linestyle='--', label='Estimated Changepoint')
plt.title('Bayesian Changepoint Detection')
plt.xlabel('Time')
plt.ylabel('Posterior Probability')
plt.legend()
plt.show()

print(f"True Changepoint: {changepoint_true}")
print(f"Estimated Changepoint: {changepoint_estimated}")