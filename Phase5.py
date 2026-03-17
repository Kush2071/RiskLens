"""
Phase 5: Stress Testing & Scenario Analysis
Scenarios: 2008 GFC · 2020 COVID Crash · 2022 Rate Hike Shock
Aligns with CCAR/ICAAP adverse scenario concepts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

PORTFOLIO_VALUE = 10_000_000
WEIGHTS         = np.array([0.15, 0.15, 0.10, 0.10, 0.10, 0.20, 0.10, 0.10])

# ─────────────────────────────────────────────
# HISTORICAL STRESS SCENARIOS
# ─────────────────────────────────────────────

SCENARIOS = {
    'Q4 Selloff 2018': {
        'start': '2018-10-01',
        'end':   '2018-12-31',
        'description': 'Fed tightening + trade war fears; S&P 500 fell ~20% in 3 months'
    },
    'COVID Crash 2020': {
        'start': '2020-02-20',
        'end':   '2020-04-30',
        'description': 'Pandemic-driven market crash (fastest 30% drop in history)'
    },
    'Rate Hike Shock 2022': {
        'start': '2022-01-01',
        'end':   '2022-12-31',
        'description': 'Fed rate hikes 0% → 4.25%; equity & bond sell-off'
    },
    'NVDA AI Drawdown 2022': {
        'start': '2022-01-01',
        'end':   '2022-10-31',
        'description': 'NVDA fell ~65% peak-to-trough; tests single-name concentration risk'
    }
}

# ─────────────────────────────────────────────
# HYPOTHETICAL SHOCK SCENARIOS
# ─────────────────────────────────────────────

HYPOTHETICAL_SHOCKS = {
    'GFC 2008 shock (hypothetical)': {
        # S&P fell ~57% peak-to-trough; applying observed avg daily returns during crisis
        # JPY and USD strengthened sharply as safe havens
        'NVDA': -0.40, 'JPM': -0.45, 'XOM': -0.30, 'JNJ': -0.15, 'SPY': -0.35,
        'EURUSD=X': -0.12, 'GBPUSD=X': -0.20, 'JPY=X': 0.15
    },
    'Equity crash –20%': {
        'NVDA': -0.35, 'JPM': -0.20, 'XOM': -0.15, 'JNJ': -0.10, 'SPY': -0.20,
        'EURUSD=X': -0.03, 'GBPUSD=X': -0.04, 'JPY=X': 0.05
    },
    'AI regulation shock': {
        'NVDA': -0.25, 'JPM': -0.02, 'XOM': 0.00, 'JNJ': 0.01, 'SPY': -0.05,
        'EURUSD=X': -0.01, 'GBPUSD=X': -0.01, 'JPY=X': 0.02
    },
    'Energy shock +30%': {
        'NVDA': -0.06, 'JPM': -0.03, 'XOM': 0.20, 'JNJ': -0.02, 'SPY': -0.04,
        'EURUSD=X': 0.01, 'GBPUSD=X': 0.00, 'JPY=X': -0.02
    }
}


# ─────────────────────────────────────────────
# HISTORICAL SCENARIO ANALYSIS
# ─────────────────────────────────────────────

def historical_scenario_analysis(dollar_pnl, log_returns, scenarios):
    """
    For each scenario window:
    - Compute cumulative P&L loss
    - Compute max drawdown
    - Compare to normal-period VaR
    """
    results = {}
    normal_var99 = -np.percentile(dollar_pnl, 1)

    for name, params in scenarios.items():
        start, end = params['start'], params['end']
        mask = (dollar_pnl.index >= start) & (dollar_pnl.index <= end)
        scenario_pnl = dollar_pnl[mask]

        if len(scenario_pnl) == 0:
            print(f"  Warning: No data for scenario '{name}'")
            continue

        cum_pnl    = scenario_pnl.sum()
        worst_day  = scenario_pnl.min()
        var_95     = -np.percentile(scenario_pnl, 5)
        var_99     = -np.percentile(scenario_pnl, 1) if len(scenario_pnl) >= 100 else np.nan
        n_days     = len(scenario_pnl)

        # Max drawdown
        cum_ret   = (scenario_pnl / PORTFOLIO_VALUE + 1).cumprod()
        roll_max  = cum_ret.cummax()
        drawdown  = (cum_ret - roll_max) / roll_max
        max_dd    = drawdown.min()

        results[name] = {
            'n_days': n_days,
            'total_pnl': cum_pnl,
            'worst_day': worst_day,
            'var_95_scenario': var_95,
            'var_99_scenario': var_99,
            'normal_var99': normal_var99,
            'var_multiplier': var_95 / normal_var99 if normal_var99 > 0 else np.nan,
            'max_drawdown_pct': max_dd * 100,
            'pnl_series': scenario_pnl
        }

    return results


# ─────────────────────────────────────────────
# HYPOTHETICAL SCENARIO P&L
# ─────────────────────────────────────────────

def hypothetical_scenario_pnl(shocks, weights, assets):
    """
    Apply instantaneous percentage shocks to each asset.
    Portfolio P&L = sum(w_i × shock_i × Portfolio Value)
    """
    results = {}
    w = pd.Series(weights, index=assets)

    for name, shock_dict in shocks.items():
        shock_series = pd.Series(shock_dict)
        # Align to our asset universe
        shock_aligned = shock_series.reindex(assets).fillna(0)
        pnl = (w * shock_aligned).sum() * PORTFOLIO_VALUE
        results[name] = pnl

    return results


# ─────────────────────────────────────────────
# PRINT RESULTS
# ─────────────────────────────────────────────

def print_stress_results(historical_results, hypo_results):
    print("\n" + "═"*65)
    print("  Historical Stress Scenario Results")
    print("═"*65)
    for name, r in historical_results.items():
        print(f"\n  {name}")
        print(f"    Period days      : {r['n_days']}")
        print(f"    Total P&L        : ${r['total_pnl']:>12,.0f}")
        print(f"    Worst single day : ${r['worst_day']:>12,.0f}")
        print(f"    Max drawdown     : {r['max_drawdown_pct']:.1f}%")
        print(f"    Scenario VaR 95% : ${r['var_95_scenario']:>12,.0f}  "
              f"({r['var_multiplier']:.1f}x normal)")

    print("\n" + "═"*65)
    print("  Hypothetical Shock Scenarios")
    print("═"*65)
    for name, pnl in hypo_results.items():
        print(f"    {name:<35}  P&L: ${pnl:>12,.0f}")


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def plot_stress_results(dollar_pnl, historical_results, hypo_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Phase 5 – Stress Testing & Scenario Analysis', fontsize=14, fontweight='bold')

    # ── Full P&L timeline with scenario shading ──
    ax = axes[0, 0]
    ax.plot(dollar_pnl.index, dollar_pnl.cumsum() / 1e6, color='steelblue', lw=1.2, label='Cumulative P&L')
    colors_scenario = ['red', 'orange', 'purple']
    for (name, params), color in zip(SCENARIOS.items(), colors_scenario):
        ax.axvspan(pd.Timestamp(params['start']), pd.Timestamp(params['end']),
                   alpha=0.15, color=color, label=name.split('(')[0].strip())
    ax.set_title('Cumulative P&L with Stress Periods')
    ax.set_ylabel('Cumulative P&L ($M)')
    ax.legend(fontsize=7)

    # ── Daily P&L comparison across scenarios ──
    ax = axes[0, 1]
    colors = ['red', 'orange', 'purple']
    for (name, r), color in zip(historical_results.items(), colors):
        pnl = r['pnl_series'] / 1e6
        ax.plot(range(len(pnl)), pnl.values, alpha=0.7, lw=1,
                label=f"{name.split('(')[0].strip()} ({len(pnl)}d)", color=color)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_title('Daily P&L During Stress Periods')
    ax.set_xlabel('Trading Days into Scenario')
    ax.set_ylabel('Daily P&L ($M)')
    ax.legend(fontsize=7)

    # ── VaR multiplier: how much worse was each scenario ──
    ax = axes[1, 0]
    names    = [n.split('(')[0].strip() for n in historical_results.keys()]
    mults    = [r['var_multiplier'] for r in historical_results.values()]
    worst    = [abs(r['worst_day']) / 1e6 for r in historical_results.values()]
    x_pos    = np.arange(len(names))
    ax.bar(x_pos - 0.2, mults, 0.35, label='Stress VaR / Normal VaR', color='tomato', alpha=0.85)
    ax2t = ax.twinx()
    ax2t.bar(x_pos + 0.2, worst, 0.35, label='Worst single day ($M)', color='steelblue', alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=8)
    ax.set_ylabel('VaR Stress Multiplier', color='tomato')
    ax2t.set_ylabel('Worst Day Loss ($M)', color='steelblue')
    ax.set_title('Stress Severity vs Normal VaR')
    ax.axhline(1.0, color='gray', linestyle='--', lw=0.8, alpha=0.7)

    # ── Hypothetical shock P&L ──
    ax = axes[1, 1]
    h_names = list(hypo_results.keys())
    h_vals  = [v / 1e6 for v in hypo_results.values()]
    bar_colors = ['tomato' if v < 0 else 'mediumseagreen' for v in h_vals]
    ax.barh(h_names, h_vals, color=bar_colors, edgecolor='white', lw=0.5)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_title('Hypothetical Shock Scenarios – Instant P&L ($M)')
    ax.set_xlabel('P&L ($M)')
    for i, v in enumerate(h_vals):
        ax.text(v + (0.01 if v >= 0 else -0.01), i,
                f'${v:.3f}M', va='center', ha='left' if v >= 0 else 'right', fontsize=8)

    plt.tight_layout()
    plt.savefig('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/phase5_stress.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Chart saved: phase5_stress.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    log_returns = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/log_returns.csv', index_col=0, parse_dates=True)
    dollar_pnl  = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/dollar_pnl.csv', index_col=0,
                               parse_dates=True).iloc[:, 0]

    w = WEIGHTS[:len(log_returns.columns)]
    w = w / w.sum()
    assets = log_returns.columns.tolist()

    historical_results = historical_scenario_analysis(dollar_pnl, log_returns, SCENARIOS)
    hypo_results       = hypothetical_scenario_pnl(HYPOTHETICAL_SHOCKS, w, assets)

    print_stress_results(historical_results, hypo_results)
    plot_stress_results(dollar_pnl, historical_results, hypo_results)
    print("\nPhase 5 complete.")