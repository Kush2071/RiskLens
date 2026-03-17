"""
Phase 4: Backtesting – Kupiec POF & Christoffersen Tests
These are the exact tests regulators use to validate internal VaR models.
Knowing these by name is a strong signal in market risk interviews.

Kupiec (1995):       Tests whether breach frequency matches expected rate.
Christoffersen(1998): Also tests whether breaches are independent (not clustered).
Basel Traffic Light:  Green / Yellow / Red zones based on breach count.
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
# BREACH IDENTIFICATION
# ─────────────────────────────────────────────

def identify_breaches(dollar_pnl, var_series):
    """
    A breach occurs on day t when the actual P&L loss exceeds
    the VaR estimated using data up to day t-1 (1-day ahead forecast).
    Here we use expanding-window VaR as the forecast.
    """
    breaches = dollar_pnl < -var_series
    return breaches.astype(int)


def rolling_var_forecast(dollar_pnl, confidence=0.99, window=252):
    """
    Rolling VaR forecast: on each day, estimate VaR from past `window` days.
    This simulates a live risk management system.
    """
    alpha   = 1 - confidence
    var_fwd = dollar_pnl.rolling(window).apply(
        lambda x: -np.percentile(x, alpha * 100), raw=True
    ).shift(1)   # shift(1) = use yesterday's estimate to forecast today
    return var_fwd


# ─────────────────────────────────────────────
# TEST 1: KUPIEC PROPORTION OF FAILURES (POF)
# ─────────────────────────────────────────────

def kupiec_pof_test(breaches, confidence):
    """
    H0: Actual failure rate p̂ = expected rate p = (1 - confidence)
    Test statistic: LR_POF ~ χ²(1) under H0

    LR_POF = -2 ln [ p^x (1-p)^(T-x) / p̂^x (1-p̂)^(T-x) ]

    where:
      T = total observations
      x = number of breaches
      p = expected failure rate (1 - confidence)
      p̂ = observed failure rate x/T
    """
    T    = len(breaches)
    x    = int(breaches.sum())
    p    = 1 - confidence       # expected failure rate
    p_hat = x / T               # observed failure rate

    if x == 0 or x == T:
        return {'x': x, 'T': T, 'p_hat': p_hat, 'p_expected': p,
                'lr_stat': np.nan, 'p_value': np.nan, 'reject_h0': False,
                'interpretation': 'Cannot compute (0 or all breaches)'}

    lr_stat = -2 * (x * np.log(p / p_hat) + (T - x) * np.log((1 - p) / (1 - p_hat)))
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
    reject  = p_value < 0.05

    return {
        'x': x, 'T': T,
        'p_hat': p_hat, 'p_expected': p,
        'lr_stat': lr_stat, 'p_value': p_value,
        'reject_h0': reject,
        'interpretation': 'FAIL – model rejected (too many/few breaches)' if reject
                          else 'PASS – failure rate is statistically acceptable'
    }


# ─────────────────────────────────────────────
# TEST 2: CHRISTOFFERSEN CONDITIONAL COVERAGE
# ─────────────────────────────────────────────

def christoffersen_test(breaches, confidence):
    """
    Extends Kupiec by also testing independence of breaches.
    A good VaR model should NOT have clustered breaches (risk underestimation in stress).

    Transition counts:
      n00 = days with no breach following no breach
      n01 = days with breach following no breach
      n10 = days with no breach following breach
      n11 = days with breach following breach  ← bad! clustering

    LR_CC = LR_POF + LR_IND ~ χ²(2)
    """
    b = breaches.values
    n00 = ((b[:-1] == 0) & (b[1:] == 0)).sum()
    n01 = ((b[:-1] == 0) & (b[1:] == 1)).sum()
    n10 = ((b[:-1] == 1) & (b[1:] == 0)).sum()
    n11 = ((b[:-1] == 1) & (b[1:] == 1)).sum()

    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi    = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x):
        return np.log(x) if x > 0 else 0

    # Likelihood ratio for independence
    L_A = (safe_log(1 - pi_01) * n00 + safe_log(pi_01) * n01 +
           safe_log(1 - pi_11) * n10 + safe_log(pi_11) * n11)
    L_0 = (safe_log(1 - pi) * (n00 + n10) + safe_log(pi) * (n01 + n11))
    lr_ind  = -2 * (L_0 - L_A)

    # Combined (POF + independence)
    kupiec  = kupiec_pof_test(breaches, confidence)
    lr_cc   = kupiec['lr_stat'] + lr_ind if not np.isnan(kupiec['lr_stat']) else np.nan
    p_value = 1 - stats.chi2.cdf(lr_cc, df=2) if not np.isnan(lr_cc) else np.nan

    reject  = p_value < 0.05 if p_value is not None and not np.isnan(p_value) else False

    return {
        'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11,
        'pi_01': pi_01, 'pi_11': pi_11,
        'lr_ind': lr_ind, 'lr_cc': lr_cc, 'p_value': p_value,
        'reject_h0': reject,
        'clustering_ratio': pi_11 / pi_01 if pi_01 > 0 else np.nan,
        'interpretation': 'FAIL – breach clustering detected' if reject
                          else 'PASS – breaches are independent (no clustering)'
    }


# ─────────────────────────────────────────────
# BASEL TRAFFIC LIGHT SYSTEM
# ─────────────────────────────────────────────

def basel_traffic_light(n_breaches, T=250):
    """
    Basel II/III uses a 250-day window at 99% confidence.
    Breach count determines regulatory capital multiplier:
      Green  (0-4):   multiplier = 3.0
      Yellow (5-9):   multiplier = 3.4 to 3.85
      Red    (10+):   multiplier = 4.0 (model may be invalidated)
    """
    if n_breaches <= 4:
        zone, multiplier = 'GREEN',  3.00
    elif n_breaches <= 9:
        add    = (n_breaches - 4) * 0.17
        zone   = 'YELLOW'
        multiplier = 3.40 + add
    else:
        zone, multiplier = 'RED', 4.00

    return zone, multiplier


# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────

def print_backtest_results(kupiec_95, kupiec_99, cc_95, cc_99, zone, mult):
    print("\n" + "═"*60)
    print("  Backtesting Results")
    print("═"*60)

    for conf_label, k, cc in [('95%', kupiec_95, cc_95), ('99%', kupiec_99, cc_99)]:
        print(f"\n  VaR Confidence: {conf_label}")
        print(f"  ── Kupiec POF Test ──")
        print(f"     Breaches:    {k['x']} / {k['T']} days")
        print(f"     Expected:    {k['p_expected']*100:.1f}% → {k['p_expected']*k['T']:.1f} breaches")
        print(f"     Observed:    {k['p_hat']*100:.2f}%")
        print(f"     LR statistic:{k['lr_stat']:.4f}  p-value: {k['p_value']:.4f}")
        print(f"     Result:      {k['interpretation']}")
        print(f"  ── Christoffersen Test ──")
        print(f"     π₀₁ (breach | no prev breach): {cc['pi_01']:.3f}")
        print(f"     π₁₁ (breach | prev breach):    {cc['pi_11']:.3f}")
        print(f"     Clustering ratio π₁₁/π₀₁:     {cc['clustering_ratio']:.2f}")
        print(f"     LR_CC: {cc['lr_cc']:.4f}  p-value: {cc['p_value']:.4f}")
        print(f"     Result: {cc['interpretation']}")

    print(f"\n  Basel Traffic Light (99% VaR, 250-day window):")
    print(f"     Breaches: {kupiec_99['x']} → Zone: {zone}  "
          f"Capital Multiplier: {mult:.2f}×")


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_backtest(dollar_pnl, var_99, breaches_99):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.suptitle('Phase 4 – Backtesting', fontsize=14, fontweight='bold')

    # ── P&L vs rolling VaR with breaches highlighted ──
    ax = axes[0]
    valid = var_99.dropna()
    pnl_aligned = dollar_pnl.loc[valid.index]
    ax.plot(pnl_aligned.index, pnl_aligned / 1e6, color='steelblue', lw=0.8, label='Daily P&L', alpha=0.7)
    ax.plot(valid.index, -valid / 1e6, color='red', lw=1.2, linestyle='--', label='1-day VaR 99%')
    breach_idx = breaches_99[breaches_99 == 1].index.intersection(pnl_aligned.index)
    ax.scatter(breach_idx, pnl_aligned.loc[breach_idx] / 1e6,
               color='red', zorder=5, s=25, label=f'Breaches ({len(breach_idx)})')
    ax.set_title('Daily P&L vs Rolling VaR (99%) – Breaches Highlighted')
    ax.set_ylabel('P&L ($M)')
    ax.legend(fontsize=8)

    # ── Cumulative breach count ──
    ax2 = axes[1]
    breach_aligned = breaches_99.loc[pnl_aligned.index]
    cum_breaches   = breach_aligned.cumsum()
    ax2.step(cum_breaches.index, cum_breaches, color='red', lw=1.5, label='Cumulative breaches')
    # Expected breach line
    expected_rate  = 0.01
    expected_cum   = pd.Series(np.arange(len(cum_breaches)) * expected_rate,
                                index=cum_breaches.index)
    ax2.plot(expected_cum.index, expected_cum, color='gray', linestyle='--',
             lw=1, label='Expected (1% rate)')
    ax2.set_title('Cumulative Breach Count vs Expected (99% VaR)')
    ax2.set_ylabel('Number of Breaches')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/phase4_backtest.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Chart saved: phase4_backtest.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    log_returns = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/log_returns.csv', index_col=0, parse_dates=True)
    dollar_pnl  = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/dollar_pnl.csv', index_col=0,
                               parse_dates=True).iloc[:, 0]

    # Rolling VaR forecasts
    var_95_rolling = rolling_var_forecast(dollar_pnl, confidence=0.95, window=252)
    var_99_rolling = rolling_var_forecast(dollar_pnl, confidence=0.99, window=252)

    # Breach series
    breaches_95 = identify_breaches(dollar_pnl, var_95_rolling).dropna()
    breaches_99 = identify_breaches(dollar_pnl, var_99_rolling).dropna()

    # Statistical tests
    kupiec_95 = kupiec_pof_test(breaches_95, 0.95)
    kupiec_99 = kupiec_pof_test(breaches_99, 0.99)
    cc_95     = christoffersen_test(breaches_95, 0.95)
    cc_99     = christoffersen_test(breaches_99, 0.99)

    # Basel traffic light
    zone, mult = basel_traffic_light(kupiec_99['x'])

    print_backtest_results(kupiec_95, kupiec_99, cc_95, cc_99, zone, mult)
    plot_backtest(dollar_pnl, var_99_rolling, breaches_99)
    print("\nPhase 4 complete.")