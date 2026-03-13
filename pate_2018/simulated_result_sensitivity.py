
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Path Setup: Point this to where your uploaded repo folder is located
# Adjust the 'base_path' if your directory structure differs.


try:
    import core as pate
    import smooth_sensitivity as pate_ss
except ImportError:
    print("Error: Could not find core.py or smooth_sensitivity.py. Check the base_path.")
    sys.exit()

# 2. Simulation Parameters (Consistent with 'Glyph' task in Table 2)
num_teachers = 5000
num_classes = 150
sigma = 100.0
order = 20.0
delta = 1e-8

# 3. Simulate Data-Dependent Privacy Metrics
agreement_pcts = np.linspace(0.1, 1.0, 50) # From 10% to 100% agreement
epsilons = []
local_sensitivities = []

for pct in agreement_pcts:
    # Create a synthetic vote histogram
    votes = np.zeros(num_classes)
    top_votes = int(pct * num_teachers)
    votes[0] = top_votes
    remaining = num_teachers - top_votes
    if remaining > 0:
        # Distribute rest of votes across other classes
        votes[1:] = remaining / (num_classes - 1)
    
    # CALL CORE.PY: Calculate log prob of incorrect outcome (Proposition 7)
    logq = pate.compute_logq_gaussian(votes, sigma)
    
    # CALL SMOOTH_SENSITIVITY.PY: Calculate RDP cost (Theorem 6)
    rdp = pate_ss._compute_rdp_gnmax(sigma, logq, order)
    
    # CALL CORE.PY: Convert RDP to standard Epsilon
    eps, _ = pate.compute_eps_from_delta([order], [rdp], delta)
    epsilons.append(eps)
    
    # CALL SMOOTH_SENSITIVITY.PY: Calculate Local Sensitivity for this query
    ls = pate_ss._compute_local_sens_gnmax(logq, sigma, num_classes, order)
    local_sensitivities.append(ls)

# 4. Generate Graphs
plt.figure(figsize=(14, 6))

# Plot 1: Privacy Cost vs. Consensus
plt.subplot(1, 2, 1)
plt.plot(agreement_pcts * 100, epsilons, color='tab:green', linewidth=3, label='Data-Dependent $\epsilon$')
plt.axhline(y=epsilons[0], color='gray', linestyle='--', label='Worst-case Baseline')
plt.title("Privacy Cost ($\epsilon$) vs. Teacher Consensus", fontsize=14)
plt.xlabel("Percentage of Teachers Agreeing (%)", fontsize=12)
plt.ylabel("Epsilon ($\epsilon$)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Local Sensitivity vs. Consensus
plt.subplot(1, 2, 2)
plt.plot(agreement_pcts * 100, local_sensitivities, color='tab:purple', linewidth=3)
plt.title("Local Sensitivity vs. Teacher Consensus", fontsize=14)
plt.xlabel("Percentage of Teachers Agreeing (%)", fontsize=12)
plt.ylabel("Local Sensitivity (LS)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("pate_paper_replication_plots.png")
plt.show()

# 5. Export results for reference
# pd.DataFrame({
#     'Agreement_Pct': agreement_pcts * 100,
#     'Epsilon': epsilons,
#     'Local_Sensitivity': local_sensitivities
# }).to_csv("simulated_pate_results.csv", index=False)