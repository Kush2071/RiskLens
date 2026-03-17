"""
Phase 3: Expected Shortfall (CVaR) – FRTB / Basel III
CVaR = Average loss given that loss exceeds VaR threshold.
Under FRTB (Fundamental Review of the Trading Book), ES replaces VaR
as the primary market risk metric. Required at 97.5% confidence.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PORTFOLIO_VALUE = 10_000_000
WEIGHTS         = np.array([0.15, 0.15, 0.10, 0.10, 0.10, 0.20, 0.10, 0.10])


# ─────────────────────────────────────────────
# CVaR FUNCTIONS
# ─────────────────────────────────────────────

def compute_cvar(dollar_pnl, confidence):
    """
    CVaR (Expected Shortfall):
      ES_α = -E[P&L | P&L ≤ -VaR_α]
    i.e. the mean of all losses beyond the VaR threshold.
    FRTB mandates 97.5% confidence.
    """
    alpha   = 1 - confidence
    var_val = -np.percentile(dollar_pnl, alpha * 100)
    tail    = dollar_pnl[dollar_pnl <= -var_val]
    es_val  = -tail.mean()
    n_obs   = len(tail)
    return var_val, es_val, n_obs


def cvar_analytical_normal(port_std, confidence):
    """
    Closed-form ES under normality:
      ES_α = σ × φ(Φ⁻¹(α)) / α × Portfolio Value
    where φ = normal PDF, Φ⁻¹ = normal quantile.
    """
    alpha = 1 - confidence
    z     = stats.norm.ppf(alpha)
    es    = stats.norm.pdf(z) / alpha * port_std * PORTFOLIO_VALUE
    return es


def rolling_cvar(dollar_pnl, confidence=0.975, window=252):
    """
    Rolling 1-year CVaR to show how tail risk evolves over time.
    Useful for understanding risk clustering (volatility regimes).
    """
    alpha = 1 - confidence
    def _es(series):
        var_t = -np.percentile(series, alpha * 100)
        tail  = series[series <= -var_t]
        return -tail.mean() if len(tail) > 0 else np.nan
    return dollar_pnl.rolling(window).apply(_es, raw=False)


# ─────────────────────────────────────────────
# CVaR DECOMPOSITION BY ASSET
# ─────────────────────────────────────────────

def component_cvar(log_returns, weights, dollar_pnl, confidence=0.975):
    """
    Marginal CVaR contribution of each asset.
    Shows which positions drive tail risk — critical for risk attribution.
    Approach: average return of each asset on the days that portfolio ES is breached.
    """
    alpha    = 1 - confidence
    var_val  = -np.percentile(dollar_pnl, alpha * 100)
    tail_idx = dollar_pnl[dollar_pnl <= -var_val].index

    w          = np.array(weights)
    tail_ret   = log_returns.loc[tail_idx]
    component  = tail_ret.mean() * w * PORTFOLIO_VALUE * -1   # average loss contribution
    pct_share  = component / component.sum() * 100

    print("\nComponent CVaR Decomposition (97.5% confidence):")
    print(f"{'Asset':<12} {'Contribution ($)':>18} {'Share (%)':>10}")
    print("-" * 42)
    for asset, contrib, pct in zip(log_returns.columns, component, pct_share):
        name = asset if len(asset) <= 10 else asset[:10]
        print(f"{name:<12} ${contrib:>15,.0f} {pct:>9.1f}%")
    return component, pct_share


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_cvar(dollar_pnl, rolling_es, component, assets, confidence_levels):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Phase 3 – Expected Shortfall (CVaR) Analysis', fontsize=14, fontweight='bold')

    # ── Loss distribution with VaR/ES shading ──
    ax = axes[0, 0]
    ax.hist(dollar_pnl / 1e6, bins=80, color='steelblue', alpha=0.7, edgecolor='white', lw=0.3)
    colors = [('orange', '--', '95%'), ('red', '--', '99%'), ('purple', '-', '97.5% (FRTB)')]
    for conf, (color, ls, lbl) in zip([0.95, 0.99, 0.975], colors):
        var_v, es_v, _ = compute_cvar(dollar_pnl, conf)
        ax.axvline(-var_v / 1e6, color=color, lw=1.5, linestyle=ls, label=f'VaR {lbl}')
        ax.axvline(-es_v / 1e6,  color=color, lw=2,   linestyle='-',  label=f'ES  {lbl}', alpha=0.6)
    # Shade tail
    threshold_97  = -compute_cvar(dollar_pnl, 0.975)[0]
    tail_vals     = dollar_pnl[dollar_pnl <= threshold_97] / 1e6
    ax.hist(tail_vals, bins=30, color='purple', alpha=0.35)
    ax.set_title('P&L Distribution with VaR & ES')
    ax.set_xlabel('Daily P&L ($M)')
    ax.legend(fontsize=7)

    # ── VaR vs ES comparison across confidence levels ──
    ax = axes[0, 1]
    confs   = np.arange(0.90, 0.9995, 0.005)
    var_arr = [-np.percentile(dollar_pnl, (1-c)*100) / 1e6 for c in confs]
    es_arr  = [compute_cvar(dollar_pnl, c)[1] / 1e6         for c in confs]
    ax.plot(confs * 100, var_arr, label='VaR',        color='steelblue', lw=1.8)
    ax.plot(confs * 100, es_arr,  label='CVaR (ES)',  color='tomato',    lw=1.8)
    ax.axvline(97.5, color='purple', linestyle='--', lw=1.2, label='97.5% (FRTB)')
    ax.set_title('VaR vs CVaR Across Confidence Levels')
    ax.set_xlabel('Confidence Level (%)')
    ax.set_ylabel('Loss ($M)')
    ax.legend(fontsize=8)

    # ── Rolling 1-year ES ──
    ax = axes[1, 0]
    ax.plot(rolling_es.index, rolling_es / 1e6, color='tomato', lw=1.2)
    ax.set_title('Rolling 1-Year CVaR (97.5%)')
    ax.set_ylabel('ES ($M)')
    ax.set_xlabel('Date')
    for event, date in [('COVID crash', '2020-03-15'), ('Rate hike shock', '2022-06-01')]:
        try:
            ax.axvline(pd.Timestamp(date), color='gray', linestyle=':', lw=1, alpha=0.7)
            ax.text(pd.Timestamp(date), rolling_es.max()/1e6 * 0.9, event,
                    rotation=45, fontsize=7, color='gray')
        except Exception:
            pass

    # ── Component CVaR bar chart ──
    ax = axes[1, 1]
    short_names = [a.replace('=X', '').replace('USD', '/USD') for a in assets]
    colors_bar  = ['steelblue' if v > 0 else 'tomato' for v in component]
    ax.barh(short_names, component / 1e6, color=colors_bar, edgecolor='white', lw=0.5)
    ax.set_title('Component CVaR by Asset (97.5%)')
    ax.set_xlabel('Contribution to ES ($M)')
    ax.axvline(0, color='black', lw=0.8)

    plt.tight_layout()
    plt.savefig('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/phase3_cvar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Chart saved: phase3_cvar.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    log_returns = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/log_returns.csv', index_col=0, parse_dates=True)
    dollar_pnl  = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/dollar_pnl.csv', index_col=0,
                               parse_dates=True).iloc[:, 0]

    w = WEIGHTS[:len(log_returns.columns)]
    w = w / w.sum()

    print("=" * 55)
    print("  CVaR / Expected Shortfall Results")
    print("=" * 55)
    for conf, label in [(0.95, '95%'), (0.975, '97.5% (FRTB)'), (0.99, '99%')]:
        var_v, es_v, n = compute_cvar(dollar_pnl, conf)
        ratio = es_v / var_v
        print(f"\n  Confidence: {label}")
        print(f"    VaR   = ${var_v:>10,.0f}")
        print(f"    ES    = ${es_v:>10,.0f}  ({ratio:.2f}x VaR)  [n={n} tail obs]")

    component, pct = component_cvar(log_returns, w, dollar_pnl, 0.975)
    rolling_es     = rolling_cvar(dollar_pnl, confidence=0.975, window=252)

    plot_cvar(dollar_pnl, rolling_es, component, log_returns.columns.tolist(),
              [0.95, 0.975, 0.99])
    print("\nPhase 3 complete.")