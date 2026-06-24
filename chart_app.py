# ============================================
# STANDALONE 15-MINUTE PROJECTION CHART
# Runs separately from Code 5
# Deploy as a separate Streamlit app
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="15-Minute Price Projection",
    page_icon="📈",
    layout="wide"
)

# --- Header ---
st.title("📈 15-Minute Price Projection Chart")
st.caption(f"Model projected price vs. actual price • Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# --- Settings ---
PREDICT_WINDOW = 15
MIN_EDGE = 0.05

COINS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD']

COIN_NAMES = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'SOL-USD': 'Solana',
    'BNB-USD': 'BNB',
    'XRP-USD': 'XRP',
    'DOGE-USD': 'Dogecoin'
}

# --- Data Fetching ---
@st.cache_data(ttl=10)
def fetch_yahoo_data(symbol, period='2d'):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval='1m')
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        return df
    except:
        return pd.DataFrame()

# --- Feature Engineering ---
def add_advanced_features(df):
    """Add comprehensive technical indicators"""
    
    df['return_1'] = df['close'].pct_change()
    df['return_5'] = df['close'].pct_change(5)
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['abs_log_return'] = df['log_return'].abs()
    
    df['lag_1'] = df['close'].shift(1)
    df['lag_5'] = df['close'].shift(5)
    
    df['volatility_5'] = df['return_1'].rolling(5).std()
    df['volatility_10'] = df['return_1'].rolling(10).std()
    df['volatility_ratio'] = df['volatility_5'] / (df['volatility_10'] + 0.001)
    
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['ema_9'] = df['close'].ewm(span=9).mean()
    df['ema_21'] = df['close'].ewm(span=21).mean()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * bb_std
    df['bb_lower'] = df['bb_middle'] - 2 * bb_std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 0.001))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    df['williams_r'] = -100 * ((high_max - df['close']) / (high_max - low_min + 0.001))
    
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad_tp = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['cci'] = (tp - sma_tp) / (0.015 * mad_tp + 0.001)
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = positive_flow / (negative_flow + 0.001)
    df['mfi'] = 100 - (100 / (1 + mfi_ratio))
    
    tr = np.maximum(df['high'] - df['low'], 
                    np.maximum(abs(df['high'] - df['close'].shift(1)), 
                              abs(df['low'] - df['close'].shift(1))))
    atr_14 = tr.rolling(14).mean()
    
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr_14)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr_14)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    df['adx'] = dx.rolling(14).mean()
    
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    
    return df

# --- Get Model Projections ---
def get_price_projections(df_clean, feature_cols, predict_window=15):
    """Generate price projections for each timestamp in the data."""
    if len(df_clean) < 100:
        return pd.DataFrame()
    
    available_cols = [col for col in feature_cols if col in df_clean.columns]
    if len(available_cols) < 10:
        return pd.DataFrame()
    
    projections = []
    
    for i in range(100, len(df_clean) - predict_window):
        train_data = df_clean.iloc[:i]
        current_data = df_clean.iloc[i]
        
        X_train = train_data[available_cols].values
        y_train = train_data['close'].shift(-predict_window) > train_data['close']
        
        X_train_df = pd.DataFrame(X_train, columns=available_cols)
        y_train = y_train.iloc[:len(X_train_df)]
        X_train_df_clean = X_train_df.dropna()
        
        if len(X_train_df_clean) < 50:
            continue
        
        X_train_clean = X_train_df_clean[available_cols].values
        y_train_clean = y_train.iloc[:len(X_train_df_clean)].values.astype(int)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_train_clean)
        
        model = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
        model.fit(X_scaled, y_train_clean)
        
        current_features = current_data[available_cols].values.reshape(1, -1)
        current_scaled = scaler.transform(current_features)
        win_prob = model.predict_proba(current_scaled)[0][1]
        
        current_price = current_data['close']
        if win_prob > 0.5:
            projected_price = current_price * (1 + (win_prob - 0.5) * 0.02)
        else:
            projected_price = current_price * (1 - (0.5 - win_prob) * 0.02)
        
        edge = win_prob - 0.50
        if edge >= MIN_EDGE and win_prob > 0.55:
            signal = "BUY YES"
        elif edge <= -MIN_EDGE and win_prob < 0.45:
            signal = "BUY NO"
        else:
            signal = "WAIT"
        
        projections.append({
            'time': current_data['time'],
            'actual_price': current_price,
            'projected_price': projected_price,
            'confidence': win_prob,
            'signal': signal,
            'edge': edge
        })
    
    return pd.DataFrame(projections)

# --- Main Chart ---
selected_coin = st.selectbox(
    "Select Coin:",
    COINS,
    format_func=lambda x: f"{COIN_NAMES.get(x, x)} ({x.replace('-USD', '')})"
)

if selected_coin:
    try:
        df = fetch_yahoo_data(selected_coin, period='2d')
        if not df.empty:
            df = add_advanced_features(df)
            df_clean = df.dropna()
            
            if len(df_clean) > 100:
                feature_cols = [
                    'close', 'volume', 'log_return', 'abs_log_return',
                    'lag_1', 'lag_5', 'volatility_5', 'volatility_10',
                    'sma_5', 'sma_10', 'rsi', 'macd_hist',
                    'bb_position', 'atr', 'stoch_k', 'stoch_d',
                    'williams_r', 'cci', 'mfi', 'adx',
                    'price_range', 'volume_ratio'
                ]
                
                projections_df = get_price_projections(df_clean, feature_cols, predict_window=15)
                
                if not projections_df.empty:
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.08,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Price Projection vs Actual", "Confidence & Signals")
                    )
                    
                    # Actual price
                    fig.add_trace(
                        go.Scatter(
                            x=projections_df['time'],
                            y=projections_df['actual_price'],
                            mode='lines',
                            name='Actual Price',
                            line=dict(color='#00b894', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Projected price
                    fig.add_trace(
                        go.Scatter(
                            x=projections_df['time'],
                            y=projections_df['projected_price'],
                            mode='lines',
                            name='Projected Price (15-min ahead)',
                            line=dict(color='#667eea', width=2, dash='dot')
                        ),
                        row=1, col=1
                    )
                    
                    # Signals
                    signal_up = projections_df[projections_df['signal'] == 'BUY YES']
                    signal_down = projections_df[projections_df['signal'] == 'BUY NO']
                    
                    if not signal_up.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=signal_up['time'],
                                y=signal_up['actual_price'],
                                mode='markers',
                                name='BUY YES Signal',
                                marker=dict(color='#00b894', size=12, symbol='triangle-up')
                            ),
                            row=1, col=1
                        )
                    
                    if not signal_down.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=signal_down['time'],
                                y=signal_down['actual_price'],
                                mode='markers',
                                name='BUY NO Signal',
                                marker=dict(color='#ff6b6b', size=12, symbol='triangle-down')
                            ),
                            row=1, col=1
                        )
                    
                    # Confidence
                    fig.add_trace(
                        go.Scatter(
                            x=projections_df['time'],
                            y=projections_df['confidence'] * 100,
                            mode='lines',
                            name='Confidence (%)',
                            line=dict(color='#fdcb6e', width=1.5)
                        ),
                        row=2, col=1
                    )
                    
                    # Edge
                    fig.add_trace(
                        go.Scatter(
                            x=projections_df['time'],
                            y=projections_df['edge'] * 100,
                            mode='lines',
                            name='Edge (%)',
                            line=dict(color='#ff6b6b', width=1.5, dash='dash')
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
                    
                    current_time = datetime.now()
                    fig.add_vline(x=current_time, line_dash="dash", line_color="white", opacity=0.5, row=1, col=1)
                    fig.add_vline(x=current_time, line_dash="dash", line_color="white", opacity=0.5, row=2, col=1)
                    
                    fig.update_layout(
                        height=600,
                        template='plotly_dark',
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                    fig.update_yaxes(title_text="%", row=2, col=1)
                    fig.update_xaxes(title_text="Time", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Stats
                    latest = projections_df.iloc[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${latest['actual_price']:.2f}")
                    col2.metric("Projected Price", f"${latest['projected_price']:.2f}")
                    col3.metric("Confidence", f"{latest['confidence']:.0%}")
                    col4.metric("Signal", latest['signal'])
                    
                    st.caption("🟢 BUY YES Signals | 🔴 BUY NO Signals | White dashed line = Current Time")
                else:
                    st.warning("Not enough data for projections. Please try again in a few minutes.")
            else:
                st.warning("Not enough data. Waiting for more price data...")
        else:
            st.warning("No data available for this coin.")
    except Exception as e:
        st.warning(f"Chart data unavailable: {e}")
else:
    st.info("Select a coin to view the projection chart.")

# --- Footer ---
st.divider()
st.caption(f"⚡ 15-Minute Price Projection • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
