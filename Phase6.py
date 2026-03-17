"""
Phase 6: Interactive Dashboard
Plotly Dash – serves at http://127.0.0.1:8050
If you prefer Tableau, export the CSVs from this script and connect them.
"""

import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

PORTFOLIO_VALUE = 10_000_000
WEIGHTS         = np.array([0.15, 0.15, 0.10, 0.10, 0.10, 0.20, 0.10, 0.10])


# ─────────────────────────────────────────────
# HELPER: COMPUTE ALL METRICS
# ─────────────────────────────────────────────

def compute_all_metrics(dollar_pnl, log_returns, w, window=252):
    alpha  = np.array([0.05, 0.025, 0.01])   # for 95%, 97.5%, 99%
    labels = ['95%', '97.5%', '99%']

    # Static VaR / ES
    static = {}
    for a, lbl in zip(alpha, labels):
        var_v  = -np.percentile(dollar_pnl, a * 100)
        tail   = dollar_pnl[dollar_pnl <= -var_v]
        es_v   = -tail.mean()
        static[lbl] = {'var': var_v, 'es': es_v}

    # Rolling VaR 99%
    roll_var = dollar_pnl.rolling(window).apply(
        lambda x: -np.percentile(x, 1), raw=True).shift(1)

    # Rolling ES 97.5% (FRTB)
    def _es(series):
        var_t = -np.percentile(series, 2.5)
        tail  = series[series <= -var_t]
        return -tail.mean() if len(tail) > 0 else np.nan
    roll_es = dollar_pnl.rolling(window).apply(_es, raw=False)

    # Breach series
    breaches = (dollar_pnl < -roll_var).astype(int)

    return static, roll_var, roll_es, breaches


# ─────────────────────────────────────────────
# BUILD PLOTLY DASHBOARD (static HTML)
# ─────────────────────────────────────────────

def build_dashboard(dollar_pnl, log_returns, w):
    static, roll_var, roll_es, breaches = compute_all_metrics(dollar_pnl, log_returns, w)

    assets = log_returns.columns.tolist()
    fig    = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            'Daily P&L vs Rolling VaR 99%',
            'P&L Distribution (Historical)',
            'Rolling CVaR 97.5% Over Time (FRTB)',
            'VaR & CVaR - Method Comparison',
            'Cumulative Breach Count',
            'Asset Contribution to Portfolio Vol'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # ── 1. Daily P&L vs VaR ──
    valid = roll_var.dropna()
    pnl_al = dollar_pnl.loc[valid.index]
    breach_idx = breaches.loc[valid.index][breaches.loc[valid.index] == 1].index

    fig.add_trace(go.Scatter(x=pnl_al.index, y=pnl_al/1e6, name='Daily P&L',
                              line=dict(color='steelblue', width=0.8), opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=valid.index, y=-valid/1e6, name='VaR 99%',
                              line=dict(color='red', width=1.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=breach_idx, y=pnl_al.loc[breach_idx]/1e6, name='Breach',
                              mode='markers', marker=dict(color='red', size=5, symbol='x')), row=1, col=1)

    # ── 2. P&L Distribution ──
    fig.add_trace(go.Histogram(x=dollar_pnl/1e6, nbinsx=80, name='P&L Dist',
                                marker_color='steelblue', opacity=0.75), row=1, col=2)
    for conf, color, lbl in [(0.95,'orange','VaR 95%'), (0.975,'purple','VaR 97.5%'), (0.99,'red','VaR 99%')]:
        var_v = -np.percentile(dollar_pnl, (1-conf)*100)
        x_pos = float(-var_v/1e6)
        fig.add_shape(type='line', x0=x_pos, x1=x_pos, y0=0, y1=1,
                      xref='x2', yref='y2 domain',
                      line=dict(color=color, dash='dash', width=1.5))
        fig.add_annotation(x=x_pos, y=0.95, xref='x2', yref='y2 domain',
                           text=lbl, showarrow=False,
                           font=dict(size=8, color=color), textangle=-45)

    # ── 3. Rolling ES ──
    fig.add_trace(go.Scatter(x=roll_es.index, y=roll_es/1e6, name='Rolling ES 97.5%',
                              line=dict(color='tomato', width=1.2), fill='tozeroy',
                              fillcolor='rgba(255,99,71,0.1)'), row=2, col=1)
    # Add vertical lines as shapes instead of add_vline (avoids plotly date parsing bug)
    for event, date in [('COVID crash', '2020-03-15'), ('Rate hike shock', '2022-06-15')]:
        fig.add_shape(type='line', x0=date, x1=date, y0=0, y1=1,
                      xref='x3', yref='y3 domain',
                      line=dict(color='gray', dash='dot', width=1))
        fig.add_annotation(x=date, y=0.95, xref='x3', yref='y3 domain',
                           text=event, showarrow=False, font=dict(size=8, color='gray'),
                           textangle=-45)

    # ── 4. Method Comparison ──
    methods = ['Historical', 'Parametric', 'Monte Carlo']
    mu  = log_returns.mean().values
    cov = log_returns.cov().values
    port_std = np.sqrt(w @ cov @ w)
    p_var95 = -stats.norm.ppf(0.05) * port_std * PORTFOLIO_VALUE
    p_var99 = -stats.norm.ppf(0.01) * port_std * PORTFOLIO_VALUE

    np.random.seed(42)
    L = np.linalg.cholesky(cov)
    Z = np.random.standard_normal((10000, len(w)))
    sim_pnl = (mu + Z @ L.T) @ w * PORTFOLIO_VALUE
    mc_var95 = -np.percentile(sim_pnl, 5)
    mc_var99 = -np.percentile(sim_pnl, 1)

    var95_vals = [static['95%']['var'], p_var95, mc_var95]
    var99_vals = [static['99%']['var'], p_var99, mc_var99]

    fig.add_trace(go.Bar(name='VaR 95%', x=methods, y=np.array(var95_vals)/1e6,
                          marker_color='steelblue', opacity=0.85), row=2, col=2)
    fig.add_trace(go.Bar(name='VaR 99%', x=methods, y=np.array(var99_vals)/1e6,
                          marker_color='tomato', opacity=0.85), row=2, col=2)

    # ── 5. Cumulative Breaches ──
    cum_breach  = breaches.loc[valid.index].cumsum()
    expected    = pd.Series(np.arange(len(cum_breach)) * 0.01, index=cum_breach.index)
    fig.add_trace(go.Scatter(x=cum_breach.index, y=cum_breach, name='Actual breaches',
                              line=dict(color='red', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=expected.index, y=expected, name='Expected (1%)',
                              line=dict(color='gray', width=1, dash='dash')), row=3, col=1)

    # ── 6. Asset Vol Contribution ──
    asset_vols  = log_returns.std() * np.sqrt(252) * 100
    label_map   = {
        'NVDA': 'NVDA', 'JPM': 'JPM', 'XOM': 'XOM',
        'JNJ': 'JNJ', 'SPY': 'SPY',
        'EURUSD=X': 'EUR/USD', 'GBPUSD=X': 'GBP/USD', 'JPYUSD=X': 'JPY/USD'
    }
    short_names = [label_map.get(a, a) for a in assets]
    fig.add_trace(go.Bar(x=short_names, y=asset_vols.values,
                          marker_color='mediumseagreen', name='Annualised Vol %',
                          opacity=0.85), row=3, col=2)

    fig.update_layout(
        title_text='Market Risk VaR Dashboard - $10M Multi-Asset Portfolio',
        title_font_size=15,
        height=1100,
        showlegend=True,
        barmode='group',
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1)
    )

    fig.update_yaxes(title_text='P&L ($M)', row=1, col=1)
    fig.update_yaxes(title_text='Frequency',   row=1, col=2)
    fig.update_yaxes(title_text='ES ($M)',     row=2, col=1)
    fig.update_yaxes(title_text='VaR ($M)',    row=2, col=2)
    fig.update_yaxes(title_text='Breaches',    row=3, col=1)
    fig.update_yaxes(title_text='Ann. Vol (%)',row=3, col=2)

    return fig


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    log_returns = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/log_returns.csv', index_col=0, parse_dates=True)
    dollar_pnl  = pd.read_csv('C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/dollar_pnl.csv', index_col=0,
                               parse_dates=True).iloc[:, 0]

    w = WEIGHTS[:len(log_returns.columns)]
    w = w / w.sum()

    print("Building dashboard...")
    fig = build_dashboard(dollar_pnl, log_returns, w)

    output_path = 'C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project/var_dashboard.html'
    fig.write_html(output_path, include_plotlyjs='cdn')
    print(f"Dashboard saved as interactive HTML: {output_path}")
    print("\nTo view: open var_dashboard.html in any browser.")
    print("To run as live Dash app, replace fig.write_html with a Dash layout.")
    print("\nPhase 6 complete.")