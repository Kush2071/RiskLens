"""
Phase 2: VaR Model Implementation – Three Methods
  1. Historical Simulation
  2. Parametric (Variance-Covariance)
  3. Monte Carlo Simulation
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

PORTFOLIO_VALUE = 10_000_000
CONFIDENCE_95   = 0.95
CONFIDENCE_99   = 0.99
N_SIMULATIONS   = 10_000
HOLDING_PERIOD  = 1   # days (set to 10 for regulatory 10-day VaR)

WEIGHTS = np.array([0.15, 0.15, 0.10, 0.10, 0.10, 0.20, 0.10, 0.10])


# ─────────────────────────────────────────────
# METHOD 1: HISTORICAL SIMULATION VaR
# ─────────────────────────────────────────────

def historical_var(dollar_pnl, confidence=0.95):
    """
    Sort historical P&L and read off the (1-conf) percentile.
    No distributional assumption — captures fat tails naturally.
    """
    alpha   = 1 - confidence
    var_val = -np.percentile(dollar_pnl, alpha * 100)
    es_val  = -dollar_pnl[dollar_pnl <= -var_val].mean()   # Expected Shortfall
    return var_val, es_val


# ─────────────────────────────────────────────
# METHOD 2: PARAMETRIC (VARIANCE-COVARIANCE) VaR
# ─────────────────────────────────────────────

def parametric_var(log_returns, weights, confidence=0.95):
    """
    Assumes normally distributed returns.
    VaR = z * σ_portfolio * Portfolio Value
    where σ_portfolio = sqrt(w' Σ w)

    Key parameters:
      μ  = mean return vector
      Σ  = covariance matrix (annualised NOT needed — use daily)
      z  = z-score at (1-conf) tail
    """
    w    = np.array(weights)
    mu   = log_returns.mean().values          # daily mean returns
    cov  = log_returns.cov().values           # daily covariance matrix

    # Portfolio mean and variance
    port_mu  = w @ mu
    port_var = w @ cov @ w
    port_std = np.sqrt(port_var)

    z     = stats.norm.ppf(1 - confidence)    # negative z for left tail
    var_val = -z * port_std * PORTFOLIO_VALUE  # flip sign → positive loss

    # Expected Shortfall (analytical formula for normal distribution)
    es_val = (stats.norm.pdf(stats.norm.ppf(1 - confidence)) /
              (1 - confidence)) * port_std * PORTFOLIO_VALUE

    return var_val, es_val, port_mu, port_std


# ─────────────────────────────────────────────
# METHOD 3: MONTE CARLO VaR
# ─────────────────────────────────────────────

def monte_carlo_var(log_returns, weights, n_sim=N_SIMULATIONS, confidence=0.95):
    """
    Simulate correlated asset returns using Cholesky decomposition.
    Steps:
      1. Compute mean vector (μ) and covariance matrix (Σ)
      2. Cholesky decompose: Σ = L L'
      3. Draw N_SIM uncorrelated standard normal vectors Z
      4. Correlated returns: R = μ + L Z
      5. Portfolio P&L = w' R * Portfolio Value
      6. Read VaR/ES from simulated distribution
    """
    w   = np.array(weights)
    mu  = log_returns.mean().values
    cov = log_returns.cov().values

    # Cholesky decomposition for correlation structure
    L = np.linalg.cholesky(cov)

    # Simulate
    np.random.seed(42)
    Z            = np.random.standard_normal((n_sim, len(w)))   # uncorrelated
    sim_returns  = mu + Z @ L.T                                  # correlated
    sim_port_ret = sim_returns @ w
    sim_pnl      = sim_port_ret * PORTFOLIO_VALUE

    alpha   = 1 - confidence
    var_val = -np.percentile(sim_pnl, alpha * 100)
    es_val  = -sim_pnl[sim_pnl <= -var_val].mean()

    return var_val, es_val, sim_pnl


# ─────────────────────────────────────────────
# SCALING TO MULTI-DAY VaR (square-root-of-time)
# ─────────────────────────────────────────────

def scale_var(var_1day, holding_period):
    """
    Regulatory standard: 10-day VaR = 1-day VaR × sqrt(10).
    Valid under IID normality assumption.
    """
    return var_1day * np.sqrt(holding_period)


# ─────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────

def print_var_summary(h_var95, h_es95, h_var99, h_es99,
                      p_var95, p_es95, p_var99, p_es99,
                      mc_var95, mc_es95, mc_var99, mc_es99):
    print("\n" + "═"*65)
    print("  VaR & Expected Shortfall Summary  (1-day, $10M portfolio)")
    print("═"*65)
    print(f"{'Method':<22} {'VaR 95%':>12} {'ES 95%':>12} {'VaR 99%':>12}  {'ES 99%':>10}")
    print("─"*65)
    fmt = lambda x: f"${x:>10,.0f}"
    print(f"{'Historical Sim':<22} {fmt(h_var95)} {fmt(h_es95)} {fmt(h_var99)} {fmt(h_es99)}")
    print(f"{'Parametric':<22} {fmt(p_var95)} {fmt(p_es95)} {fmt(p_var99)} {fmt(p_es99)}")
    print(f"{'Monte Carlo':<22} {fmt(mc_var95)} {fmt(mc_es95)} {fmt(mc_var99)} {fmt(mc_es99)}")
    print("═"*65)
    print(f"\n  10-day VaR (regulatory, 99%, Historical): "
          f"${scale_var(h_var99, 10):>10,.0f}")
    print(f"  10-day VaR (regulatory, 99%, Monte Carlo): "
          f"${scale_var(mc_var99, 10):>10,.0f}")


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_var_comparison(dollar_pnl, mc_pnl,
                        h_var95, h_var99,
                        p_var95, p_var99,
                        mc_var95, mc_var99):

    fig = plt.figure(figsize=(15, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle('Phase 2 – VaR Model Comparison', fontsize=14, fontweight='bold')

    # ── Historical P&L distribution ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(dollar_pnl / 1e6, bins=80, color='steelblue', edgecolor='white', lw=0.3, alpha=0.85)
    ax1.axvline(-h_var95 / 1e6, color='orange',  lw=2, linestyle='--', label='VaR 95%')
    ax1.axvline(-h_var99 / 1e6, color='red',     lw=2, linestyle='--', label='VaR 99%')
    ax1.set_title('Historical Simulation')
    ax1.set_xlabel('Daily P&L ($M)')
    ax1.legend(fontsize=8)

    # ── Parametric normal overlay ──
    ax2 = fig.add_subplot(gs[0, 1])
    mu_p  = dollar_pnl.mean()
    std_p = dollar_pnl.std()
    x     = np.linspace(dollar_pnl.min(), dollar_pnl.max(), 300)
    ax2.hist(dollar_pnl / 1e6, bins=80, density=True, color='steelblue', alpha=0.6, edgecolor='white', lw=0.3)
    ax2.plot(x / 1e6, stats.norm.pdf(x, mu_p, std_p) * 1e6, 'k-', lw=1.8, label='Normal fit')
    ax2.axvline(-p_var95 / 1e6, color='orange', lw=2, linestyle='--', label='VaR 95%')
    ax2.axvline(-p_var99 / 1e6, color='red',    lw=2, linestyle='--', label='VaR 99%')
    ax2.set_title('Parametric (Normal)')
    ax2.set_xlabel('Daily P&L ($M)')
    ax2.legend(fontsize=8)

    # ── Monte Carlo distribution ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(mc_pnl / 1e6, bins=100, color='mediumseagreen', edgecolor='white', lw=0.3, alpha=0.85)
    ax3.axvline(-mc_var95 / 1e6, color='orange', lw=2, linestyle='--', label='VaR 95%')
    ax3.axvline(-mc_var99 / 1e6, color='red',    lw=2, linestyle='--', label='VaR 99%')
    ax3.set_title(f'Monte Carlo ({N_SIMULATIONS:,} sims)')
    ax3.set_xlabel('Daily P&L ($M)')
    ax3.legend(fontsize=8)

    # ── Method comparison bar chart ──
    ax4 = fig.add_subplot(gs[1, :2])
    methods = ['Historical', 'Parametric', 'Monte Carlo']
    var95_vals = [h_var95, p_var95, mc_var95]
    var99_vals = [h_var99, p_var99, mc_var99]
    x_pos = np.arange(len(methods))
    w     = 0.35
    ax4.bar(x_pos - w/2, np.array(var95_vals) / 1e6, w, label='VaR 95%', color='steelblue', alpha=0.85)
    ax4.bar(x_pos + w/2, np.array(var99_vals) / 1e6, w, label='VaR 99%', color='tomato',    alpha=0.85)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(methods)
    ax4.set_ylabel('VaR ($M)')
    ax4.set_title('VaR Comparison Across Methods')
    ax4.legend()
    for i, (v95, v99) in enumerate(zip(var95_vals, var99_vals)):
        ax4.text(i - w/2, v95/1e6 + 0.01, f'${v95/1e6:.3f}M', ha='center', fontsize=8)
        ax4.text(i + w/2, v99/1e6 + 0.01, f'${v99/1e6:.3f}M', ha='center', fontsize=8)

    # ── QQ plot to test normality assumption ──
    ax5 = fig.add_subplot(gs[1, 2])
    stats.probplot(dollar_pnl / 1e6, dist='norm', plot=ax5)
    ax5.set_title('Q-Q Plot (normality check)')
    ax5.get_lines()[0].set(markersize=2, alpha=0.5)

    plt.savefig('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/phase2_var_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Chart saved: phase2_var_comparison.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    log_returns = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/log_returns.csv', index_col=0, parse_dates=True)
    dollar_pnl  = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/dollar_pnl.csv', index_col=0,
                               parse_dates=True).iloc[:, 0]

    # Align weights
    w = WEIGHTS[:len(log_returns.columns)]
    w = w / w.sum()

    # Method 1: Historical
    h_var95, h_es95 = historical_var(dollar_pnl, 0.95)
    h_var99, h_es99 = historical_var(dollar_pnl, 0.99)

    # Method 2: Parametric
    p_var95, p_es95, mu, std = parametric_var(log_returns, w, 0.95)
    p_var99, p_es99, _,   _  = parametric_var(log_returns, w, 0.99)

    # Method 3: Monte Carlo
    mc_var95, mc_es95, mc_pnl = monte_carlo_var(log_returns, w, N_SIMULATIONS, 0.95)
    mc_var99, mc_es99, _      = monte_carlo_var(log_returns, w, N_SIMULATIONS, 0.99)

    print_var_summary(h_var95, h_es95, h_var99, h_es99,
                      p_var95, p_es95, p_var99, p_es99,
                      mc_var95, mc_es95, mc_var99, mc_es99)

    plot_var_comparison(dollar_pnl, mc_pnl,
                        h_var95, h_var99,
                        p_var95, p_var99,
                        mc_var95, mc_var99)

    # Save results for phase 4
    results = {
        'h_var95': h_var95, 'h_var99': h_var99,
        'p_var95': p_var95, 'p_var99': p_var99,
        'mc_var95': mc_var95, 'mc_var99': mc_var99
    }
    pd.Series(results).to_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/var_results.csv')
    print("\nPhase 2 complete.")