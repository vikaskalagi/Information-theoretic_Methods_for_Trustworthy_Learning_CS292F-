import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure path to core.py is correct
# sys.path.append('tensorflow/privacy/privacy-26bc5cb10d08a3ccd208af3db01427f8330b51c0/research/pate_2018')
import core as pate

def simulate_iclr_graph():
    num_teachers = 250
    max_queries = 1000
    delta = 1e-8
    
    # Noise scales typical for the paper's benchmarks
    lambda_laplace = 50.0  # Scale for LNMax
    sigma_gaussian = 40.0  # Scale for GNMax
    
    # RDP orders for the accountant
    orders = np.concatenate((np.arange(2, 100 + 1, .5),
                             np.logspace(np.log10(100), np.log10(500), num=100)))

    # --- Simulated High Consensus Votes ---
    # To simulate the "Scalable" claim, we assume teachers agree well (200/250)
    # votes_sample = np.zeros(10, dtype=int)
    # votes_sample[0], votes_sample[1] = 200, 50
    
    votes_sample = np.zeros(10, dtype=int)
    votes_sample[0] = 245  
    votes_sample[1] = 5

    x_axis = np.arange(1, max_queries + 1)
    y_lnmax = []
    y_gnmax = []

    # 1. Compute LNMax (Laplace) Cumulative Privacy
    # For LNMax, the RDP is calculated using rdp_pure_eps
    logq_lap = pate.compute_logq_laplace(votes_sample, lambda_laplace)
    rdp_query_lap = pate.rdp_pure_eps(logq_lap, 2.0 / lambda_laplace, orders)
    
    # votes_high = np.zeros(10, dtype=int)
    # votes_high[0] = 245  
    # votes_high[1] = 5
    
    # 2. Compute GNMax (Gaussian) Cumulative Privacy
    logq_gauss = pate.compute_logq_gaussian(votes_sample, sigma_gaussian)
    rdp_query_gauss = pate.rdp_gaussian(logq_gauss, sigma_gaussian, orders)

    for q in x_axis:
        # Cumulative cost: sum of RDP for 'q' queries
        eps_lap, _ = pate.compute_eps_from_delta(orders, rdp_query_lap * q, delta)
        eps_gauss, _ = pate.compute_eps_from_delta(orders, rdp_query_gauss * q, delta)
        
        y_lnmax.append(eps_lap)
        y_gnmax.append(eps_gauss)

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, y_lnmax, color='r', linestyle='--', label='LNMax (Laplace)', linewidth=2.5)
    plt.plot(x_axis, y_gnmax, color='g', linestyle='-', label='Confident-GNMax (Gaussian)', linewidth=2.5)
    
    plt.title("Cumulative Privacy Cost: LNMax vs. Confident-GNMax")
    plt.xlabel("Number of queries answered")
    plt.ylabel(r"Privacy cost $\varepsilon$ at $\delta=10^{-8}$")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pate_laplace_vs_gaussian.png')
    plt.show()

if __name__ == "__main__":
    simulate_iclr_graph()