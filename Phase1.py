"""
Phase 1: Data Collection & Portfolio Construction

Equities  -> stooq.com via pandas_datareader
FX pairs  -> FRED (Federal Reserve) via pandas_datareader

All 3 FX pairs are USD-denominated (USD per 1 foreign currency):
  EUR/USD = USD per 1 EUR
  GBP/USD = USD per 1 GBP
  JPY/USD = USD per 1 JPY  (FRED DEXJPUS inverted)

Install once in Spyder terminal:
    pip install pandas_datareader
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pandas_datareader import data as pdr

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

FOLDER = r'C:/Users/Kush Bhanushali/OneDrive/Desktop/New Project'

# Equities from stooq.com
STOOQ_TICKERS = {
    'NVDA': 'NVDA.US',   # Technology / AI
    'JPM':  'JPM.US',    # Financials
    'XOM':  'XOM.US',    # Energy
    'JNJ':  'JNJ.US',    # Healthcare
    'SPY':  'SPY.US',    # Broad Market ETF
}

# FX from FRED — all converted to USD per 1 foreign currency
# DEXUSEU = USD per 1 EUR  -> already correct
# DEXUSUK = USD per 1 GBP  -> already correct
# DEXJPUS = JPY per 1 USD  -> we INVERT to get USD per 1 JPY
FRED_FX = {
    'EURUSD=X': 'DEXUSEU',
    'GBPUSD=X': 'DEXUSUK',
    'JPYUSD=X': 'DEXJPUS',   # will be inverted below
}

FX_LABELS = {
    'EURUSD=X': 'EUR/USD',
    'GBPUSD=X': 'GBP/USD',
    'JPYUSD=X': 'JPY/USD',   # inverted — USD per 1 JPY
}

# Portfolio weights (must sum to 1)
# Order: NVDA, JPM, XOM, JNJ, SPY, EURUSD, GBPUSD, JPYUSD
WEIGHTS = np.array([0.15, 0.15, 0.10, 0.10, 0.10,
                    0.20, 0.10, 0.10])

assert abs(WEIGHTS.sum() - 1.0) < 1e-9, "Weights must sum to 1"

PORTFOLIO_VALUE = 10_000_000
START_DATE      = '2015-01-01'
END_DATE        = '2024-12-31'


# ─────────────────────────────────────────────
# 1. DOWNLOAD EQUITIES FROM STOOQ
# ─────────────────────────────────────────────

def download_equities(tickers_map, start, end):
    import time
    prices_list = {}

    print("=" * 55)
    print("  EQUITIES — stooq.com")
    print("=" * 55)

    for ticker, stooq_ticker in tickers_map.items():
        try:
            print(f"  Downloading {ticker} ({stooq_ticker})...")
            df = pdr.DataReader(stooq_ticker, 'stooq', start=start, end=end)
            df = df.sort_index(ascending=True)

            if len(df) > 0:
                prices_list[ticker] = df['Close']
                print(f"    OK: {len(df)} trading days")
            else:
                print(f"    FAILED: no data returned")

            time.sleep(0.5)

        except Exception as e:
            print(f"    FAILED: {e}")

    return pd.DataFrame(prices_list)


# ─────────────────────────────────────────────
# 2. DOWNLOAD FX FROM FRED
# ─────────────────────────────────────────────

def download_fx(fred_map, start, end):
    import time
    fx_list = {}

    print("\n" + "=" * 55)
    print("  FX PAIRS — FRED (Federal Reserve)")
    print("=" * 55)

    for ticker, fred_code in fred_map.items():
        try:
            print(f"  Downloading {FX_LABELS[ticker]} (FRED: {fred_code})...")
            df = pdr.DataReader(fred_code, 'fred', start=start, end=end)
            df = df.dropna()

            if len(df) > 0:
                series = df[fred_code]

                # Invert DEXJPUS (JPY per USD) -> USD per JPY
                # so all 3 pairs are consistently USD per 1 foreign currency
                if fred_code == 'DEXJPUS':
                    series = 1 / series
                    print(f"    Inverted: JPY per USD -> USD per JPY")

                fx_list[ticker] = series
                print(f"    OK: {len(df)} observations")
            else:
                print(f"    FAILED: no data returned")

            time.sleep(0.5)

        except Exception as e:
            print(f"    FAILED: {e}")

    return pd.DataFrame(fx_list)


# ─────────────────────────────────────────────
# 3. MERGE & ALIGN
# ─────────────────────────────────────────────

def merge_prices(equity_prices, fx_prices, start, end):
    """
    Merge equity (stooq) and FX (FRED) on common trading days.
    Forward-fill FX gaps up to 3 days to handle minor holiday mismatches.
    """
    print("\n" + "=" * 55)
    print("  MERGING & ALIGNING")
    print("=" * 55)

    fx_filled = fx_prices.ffill(limit=3)

    combined = equity_prices.copy()
    for col in fx_filled.columns:
        combined[col] = fx_filled[col].reindex(equity_prices.index, method='ffill')

    combined = combined[(combined.index >= start) & (combined.index <= end)]
    combined = combined.dropna()

    print(f"  Combined assets   : {list(combined.columns)}")
    print(f"  Trading days      : {len(combined)}")
    print(f"  Date range        : {combined.index[0].date()} to {combined.index[-1].date()}")

    return combined


# ─────────────────────────────────────────────
# 4. COMPUTE RETURNS
# ─────────────────────────────────────────────

def compute_returns(prices):
    log_returns = np.log(prices / prices.shift(1)).dropna()

    print(f"\n{'Asset':<12} {'Ann.Return%':>12} {'Ann.Vol%':>10} {'Skew':>8} {'Kurt':>8}")
    print("-" * 55)
    for col in log_returns.columns:
        r    = log_returns[col]
        name = FX_LABELS.get(col, col)
        print(f"{name:<12} {r.mean()*252*100:>11.2f}%"
              f" {r.std()*np.sqrt(252)*100:>9.2f}%"
              f"  {r.skew():>6.2f}  {r.kurtosis():>6.2f}")
    return log_returns


# ─────────────────────────────────────────────
# 5. PORTFOLIO RETURNS
# ─────────────────────────────────────────────

def compute_portfolio_returns(log_returns, weights):
    w            = np.array(weights[:len(log_returns.columns)])
    w            = w / w.sum()
    port_returns = log_returns.dot(w)
    dollar_pnl   = port_returns * PORTFOLIO_VALUE

    print(f"\nPortfolio summary:")
    print(f"  Annualised return : {port_returns.mean()*252*100:.2f}%")
    print(f"  Annualised vol    : {port_returns.std()*np.sqrt(252)*100:.2f}%")
    print(f"  Sharpe ratio      : {port_returns.mean()/port_returns.std()*np.sqrt(252):.2f}")
    print(f"  Max daily loss    : ${dollar_pnl.min():,.0f}")
    print(f"  Max daily gain    : ${dollar_pnl.max():,.0f}")
    return port_returns, dollar_pnl


# ─────────────────────────────────────────────
# 6. VISUALISE
# ─────────────────────────────────────────────

def plot_portfolio(prices, dollar_pnl):
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Portfolio Overview - Phase 1 (2015-2024)',
                 fontsize=15, fontweight='bold', y=1.01)

    colors_eq = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_fx = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # ── Top row: Equities log scale ──
    ax1 = fig.add_subplot(3, 2, (1, 2))
    eq_cols = [c for c in STOOQ_TICKERS.keys() if c in prices.columns]
    norm    = prices[eq_cols] / prices[eq_cols].iloc[0] * 100
    for col, color in zip(norm.columns, colors_eq):
        ax1.plot(norm.index, norm[col], label=col, linewidth=1.4, color=color)
    ax1.set_yscale('log')
    ax1.set_title('Equities - Normalised Price (base=100, log scale)', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_ylabel('Index (log scale)')
    ax1.grid(axis='y', alpha=0.3)

    # ── Middle row: one subplot per FX pair (all USD-denominated now) ──
    fx_cols = [c for c in FRED_FX.keys() if c in prices.columns]
    for i, (col, color) in enumerate(zip(fx_cols, colors_fx)):
        ax = fig.add_subplot(3, 3, 4 + i)
        ax.plot(prices.index, prices[col], color=color, linewidth=1.2)
        ax.set_title(FX_LABELS[col] + ' — USD per 1 unit', fontsize=9)
        ax.set_ylabel('Rate')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        ax.grid(axis='y', alpha=0.3)

        # Annotate direction: all go DOWN when USD strengthens
        ax.text(0.02, 0.05, 'All pairs: USD per 1 foreign unit\n↓ = USD strengthens',
                transform=ax.transAxes, fontsize=6.5, color='gray',
                verticalalignment='bottom')

    # ── Bottom left: P&L distribution ──
    ax4 = fig.add_subplot(3, 2, 5)
    ax4.hist(dollar_pnl / 1e6, bins=80, color='steelblue',
             edgecolor='white', linewidth=0.3)
    ax4.axvline(dollar_pnl.quantile(0.05) / 1e6, color='red',
                linestyle='--', linewidth=1.5,
                label=f'VaR 95%: ${abs(dollar_pnl.quantile(0.05)/1e6):.3f}M')
    ax4.axvline(dollar_pnl.quantile(0.01) / 1e6, color='darkred',
                linestyle='--', linewidth=1.5,
                label=f'VaR 99%: ${abs(dollar_pnl.quantile(0.01)/1e6):.3f}M')
    ax4.set_title('Daily Portfolio P&L Distribution', fontsize=11)
    ax4.set_xlabel('P&L ($M)')
    ax4.legend(fontsize=8)
    ax4.grid(axis='y', alpha=0.3)

    # ── Bottom right: Cumulative portfolio value ──
    ax5 = fig.add_subplot(3, 2, 6)
    cum = (dollar_pnl / PORTFOLIO_VALUE + 1).cumprod() * PORTFOLIO_VALUE
    ax5.plot(cum.index, cum / 1e6, color='steelblue', linewidth=1.4)
    ax5.set_title('Cumulative Portfolio Value ($M)', fontsize=11)
    ax5.set_ylabel('$M')
    ax5.grid(axis='y', alpha=0.3)
    for event, date, offset in [('COVID\ncrash',    '2020-03-20', 1.5),
                                  ('Rate hike\nshock', '2022-10-01', 1.5)]:
        try:
            ax5.axvline(pd.Timestamp(date), color='gray', linestyle=':', linewidth=1)
            ax5.text(pd.Timestamp(date), cum.min() / 1e6 + offset,
                     event, fontsize=7, color='gray', ha='center')
        except Exception:
            pass

    plt.tight_layout()
    plt.savefig(FOLDER + '/phase1_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nChart saved: {FOLDER}/phase1_overview.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    equity_prices = download_equities(STOOQ_TICKERS, START_DATE, END_DATE)
    fx_prices     = download_fx(FRED_FX, START_DATE, END_DATE)

    if equity_prices.shape[1] == 0:
        print("\nERROR: No equity data. Check stooq connection.")
    elif fx_prices.shape[1] == 0:
        print("\nERROR: No FX data. Check FRED connection.")
    else:
        prices = merge_prices(equity_prices, fx_prices, START_DATE, END_DATE)

        print("\n" + "=" * 55)
        print("  RETURN STATISTICS (annualised)")
        print("=" * 55)
        log_returns              = compute_returns(prices)
        port_returns, dollar_pnl = compute_portfolio_returns(log_returns, WEIGHTS)
        plot_portfolio(prices, dollar_pnl)

        log_returns.to_csv(FOLDER + '/log_returns.csv')
        dollar_pnl.to_csv( FOLDER + '/dollar_pnl.csv', header=True)
        print(f"\nSaved: log_returns.csv and dollar_pnl.csv to {FOLDER}")
        print("Phase 1 complete — now run phase2_var_models.py")