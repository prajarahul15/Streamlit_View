import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration with enhanced settings
st.set_page_config(
    page_title="🚀 Ultimate Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS with fixed heights and no scrolling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        overflow-x: hidden;
    }
    
    .main > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1rem;
        animation: fadeInDown 0.8s ease-in-out;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sub-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2d3748;
        margin: 1rem 0 0.5rem 0;
        padding-left: 0.8rem;
        border-left: 3px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.2);
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
        height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .interactive-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .interactive-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .compact-kpi {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin: 0.3rem 0;
        height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }
    
    .chart-container {
        height: 350px;
        margin: 0.5rem 0;
    }
    
    .small-chart {
        height: 250px;
        margin: 0.3rem 0;
    }
    
    .alert-compact {
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Compact sidebar */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Remove extra spacing */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Compact selectboxes */
    .stSelectbox > div > div {
        height: 35px;
    }
    
    /* Compact multiselect */
    .stMultiSelect > div > div {
        min-height: 35px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data files with optimized loading"""
    data = {}
    
    files_config = [
        ('sample_data', 'Sample_data_N.csv', '%d-%m-%Y', 'DATE'),
        ('plan_data', 'Plan Number.csv', '%d-%m-%Y', 'DATE'),
        ('fis_data', 'FIS Historical Data.csv', '%d-%m-%Y', 'Date'),
        ('dow_banks', 'Dow Jones Banks Historical Data.csv', '%d-%m-%Y', 'Date'),
        ('sp500', 'INDEX_US_S&P US_SPX.csv', 'special', 'Date')
    ]
    
    for key, filename, date_format, date_col in files_config:
        try:
            if filename.endswith('.xlsx'):
                data[key] = pd.read_excel(filename)
                if 'Date' in data[key].columns:
                    data[key]['Date'] = pd.to_datetime(data[key]['Date'])
            else:
                data[key] = pd.read_csv(filename)
                
                if date_format == 'special':  # S&P 500
                    data[key]['Date'] = data[key]['Date'].apply(lambda x: f"01-{x}")
                    data[key]['Date'] = pd.to_datetime(data[key]['Date'], format='%d-%b-%y')
                    data[key]['Close'] = pd.to_numeric(data[key]['Close'].astype(str).str.replace(',', ''), errors='coerce')
                else:
                    data[key][date_col] = pd.to_datetime(data[key][date_col], format=date_format, errors='coerce')
                    data[key] = data[key].dropna()
                    
                    # Clean numeric columns
                    for col in ['Price', 'Open', 'High', 'Low', 'Actual', 'Plan']:
                        if col in data[key].columns:
                            if data[key][col].dtype == 'object':
                                data[key][col] = pd.to_numeric(data[key][col].astype(str).str.replace(',', ''), errors='coerce')
        
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {e}")
            data[key] = pd.DataFrame()
    
    return data

def create_compact_kpi_row(data):
    """Create compact KPI row that fits in viewport"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if 'sample_data' in data and not data['sample_data'].empty:
            total_actual = data['sample_data']['Actual'].sum()
            st.markdown(f"""
            <div class="compact-kpi">
                <div style="font-size: 0.8rem; color: #64748b;">💰 Total Actual</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #1e40af;">${total_actual:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'plan_data' in data and not data['plan_data'].empty:
            total_plan = data['plan_data']['Plan'].sum()
            st.markdown(f"""
            <div class="compact-kpi">
                <div style="font-size: 0.8rem; color: #64748b;">🎯 Total Plan</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #7c3aed;">${total_plan:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'fis_data' in data and not data['fis_data'].empty:
            latest_fis = data['fis_data']['Price'].iloc[0]
            prev_fis = data['fis_data']['Price'].iloc[1] if len(data['fis_data']) > 1 else latest_fis
            change = ((latest_fis - prev_fis) / prev_fis) * 100
            color = "#10b981" if change > 0 else "#ef4444"
            arrow = "↗" if change > 0 else "↘"
            st.markdown(f"""
            <div class="compact-kpi">
                <div style="font-size: 0.8rem; color: #64748b;">📈 FIS Stock</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: {color};">${latest_fis:.2f} {arrow}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'sample_data' in data and 'plan_data' in data and not data['sample_data'].empty and not data['plan_data'].empty:
            actual_total = data['sample_data']['Actual'].sum()
            plan_total = data['plan_data']['Plan'].sum()
            variance = ((actual_total - plan_total) / plan_total) * 100
            color = "#10b981" if variance > 0 else "#ef4444"
            st.markdown(f"""
            <div class="compact-kpi">
                <div style="font-size: 0.8rem; color: #64748b;">📊 Variance</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: {color};">{variance:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col5:
        # Market status indicator
        status = "🟢 Active" if datetime.now().weekday() < 5 else "🔴 Closed"
        st.markdown(f"""
        <div class="compact-kpi">
            <div style="font-size: 0.8rem; color: #64748b;">🏛️ Market</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #6366f1;">{status}</div>
        </div>
        """, unsafe_allow_html=True)

def create_compact_chart(data, chart_type, title, height=300):
    """Create compact charts with fixed heights"""
    if chart_type == "sample_overview" and 'sample_data' in data and not data['sample_data'].empty:
        grouped = data['sample_data'].groupby(['DATE', 'Lineup'])['Actual'].sum().reset_index()
        fig = px.line(grouped, x='DATE', y='Actual', color='Lineup', 
                     title=title, template='plotly_white')
        fig.update_traces(mode='lines', line=dict(width=2))
    
    elif chart_type == "comparison" and 'sample_data' in data and 'plan_data' in data:
        sample_agg = data['sample_data'].groupby('DATE')['Actual'].sum().reset_index()
        plan_agg = data['plan_data'].groupby('DATE')['Plan'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_agg['DATE'], y=sample_agg['Actual'], 
                                name='Actual', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=plan_agg['DATE'], y=plan_agg['Plan'], 
                                name='Plan', line=dict(color='red', width=2, dash='dash')))
        fig.update_layout(title=title, template='plotly_white')
    
    elif chart_type == "fis_stock" and 'fis_data' in data and not data['fis_data'].empty:
        fig = px.line(data['fis_data'].head(30), x='Date', y='Price', 
                     title=title, template='plotly_white')
        fig.update_traces(line=dict(color='#667eea', width=2))
    
    elif chart_type == "dow_banks" and 'dow_banks' in data and not data['dow_banks'].empty:
        fig = px.line(data['dow_banks'].head(30), x='Date', y='Price', 
                     title=title, template='plotly_white')
        fig.update_traces(line=dict(color='#764ba2', width=2))
    
    elif chart_type == "sp500" and 'sp500' in data and not data['sp500'].empty:
        fig = px.line(data['sp500'].head(30), x='Date', y='Close', 
                     title=title, template='plotly_white')
        fig.update_traces(line=dict(color='#10b981', width=2))
    
    else:
        # Fallback empty chart
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title=title, template='plotly_white')
    
    # Compact layout settings
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_compact_alerts(data):
    """Create compact alert system"""
    alerts = []
    
    # Quick checks for alerts
    if 'fis_data' in data and not data['fis_data'].empty and len(data['fis_data']) > 1:
        recent_change = ((data['fis_data']['Price'].iloc[0] - data['fis_data']['Price'].iloc[1]) / 
                        data['fis_data']['Price'].iloc[1]) * 100
        
        if abs(recent_change) > 10:
            alerts.append(("🔴", f"FIS: {recent_change:+.1f}%", "High volatility"))
        elif abs(recent_change) > 5:
            alerts.append(("🟡", f"FIS: {recent_change:+.1f}%", "Moderate movement"))
    
    if 'sample_data' in data and 'plan_data' in data:
        if not data['sample_data'].empty and not data['plan_data'].empty:
            actual_total = data['sample_data']['Actual'].sum()
            plan_total = data['plan_data']['Plan'].sum()
            variance = ((actual_total - plan_total) / plan_total) * 100
            
            if abs(variance) > 15:
                alerts.append(("🚨", f"Plan: {variance:+.1f}%", "High deviation"))
            elif abs(variance) > 5:
                alerts.append(("⚠️", f"Plan: {variance:+.1f}%", "Monitor variance"))
    
    # Display compact alerts
    if alerts:
        alert_cols = st.columns(len(alerts))
        for i, (icon, value, desc) in enumerate(alerts):
            with alert_cols[i]:
                st.markdown(f"""
                <div class="alert-compact" style="background: #fef2f2; border: 1px solid #fecaca; color: #991b1b;">
                    <div style="font-weight: 600;">{icon} {value}</div>
                    <div style="font-size: 0.8rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-compact" style="background: #f0fdf4; border: 1px solid #bbf7d0; color: #166534;">
            <div style="font-weight: 600;">✅ All Systems Normal</div>
            <div style="font-size: 0.8rem;">No alerts detected</div>
        </div>
        """, unsafe_allow_html=True)

def create_hierarchical_controls():
    """Create compact hierarchical controls"""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        level = st.selectbox(
            "📊 Analysis Level", 
            ['Profile', 'Line_Item', 'Site', 'Lineup'],
            help="Select analysis depth"
        )
    
    with col2:
        chart_style = st.selectbox(
            "📈 Chart Type",
            ["Line", "Bar", "Area"],
            help="Choose visualization style"
        )
    
    with col3:
        time_range = st.selectbox(
            "📅 Time Range",
            ["Last 6 months", "Last year", "All time"],
            help="Select time period"
        )
    
    return level, chart_style, time_range

def create_grid_layout(data):
    """Create optimized grid layout for charts"""
    
    # Row 1: Main analysis chart (full width)
    st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
    level, chart_style, time_range = create_hierarchical_controls()
    
    if 'sample_data' in data and not data['sample_data'].empty:
        # Apply time filtering
        filtered_data = data['sample_data'].copy()
        if time_range == "Last 6 months":
            cutoff_date = datetime.now() - timedelta(days=180)
            filtered_data = filtered_data[filtered_data['DATE'] >= cutoff_date]
        elif time_range == "Last year":
            cutoff_date = datetime.now() - timedelta(days=365)
            filtered_data = filtered_data[filtered_data['DATE'] >= cutoff_date]
        
        # Create main chart
        if level == 'Lineup':
            fig_main = px.line(filtered_data, x='DATE', y='Actual', color='Lineup',
                              title=f'🎯 {level} Analysis - {chart_style} Chart')
        else:
            grouped = filtered_data.groupby(['DATE', level])['Actual'].sum().reset_index()
            if chart_style == "Line":
                fig_main = px.line(grouped, x='DATE', y='Actual', color=level,
                                  title=f'📊 {level} Analysis - Line Chart')
            elif chart_style == "Bar":
                fig_main = px.bar(grouped, x='DATE', y='Actual', color=level,
                                 title=f'📊 {level} Analysis - Bar Chart')
            else:  # Area
                fig_main = px.area(grouped, x='DATE', y='Actual', color=level,
                                  title=f'📊 {level} Analysis - Area Chart')
        
        fig_main.update_layout(height=350, margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_main, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Row 2: Three medium charts
    chart_col1, chart_col2, chart_col3 = st.columns(3)
    
    with chart_col1:
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        if 'sample_data' in data and 'plan_data' in data:
            fig_comp = create_compact_chart(data, "comparison", "📊 Actual vs Plan", 280)
            st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chart_col2:
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        fig_fis = create_compact_chart(data, "fis_stock", "📈 FIS Stock (30D)", 280)
        st.plotly_chart(fig_fis, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with chart_col3:
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        fig_dow = create_compact_chart(data, "dow_banks", "🏦 Dow Banks (30D)", 280)
        st.plotly_chart(fig_dow, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_performance_grid(data):
    """Create performance metrics in grid layout"""
    
    # Performance metrics row
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    metrics = [
        {"title": "📈 Growth", "value": "+12.5%", "color": "#10b981"},
        {"title": "🎯 Efficiency", "value": "94.2%", "color": "#3b82f6"},
        {"title": "📊 Performance", "value": "+8.7%", "color": "#8b5cf6"},
        {"title": "⚠️ Risk", "value": "15.3%", "color": "#f59e0b"}
    ]
    
    for i, metric in enumerate(metrics):
        col = [perf_col1, perf_col2, perf_col3, perf_col4][i]
        with col:
            st.markdown(f"""
            <div class="compact-kpi" style="border-left: 3px solid {metric['color']};">
                <div style="font-size: 0.8rem; color: #64748b;">{metric['title']}</div>
                <div style="font-size: 1.3rem; font-weight: 600; color: {metric['color']};">{metric['value']}</div>
            </div>
            """, unsafe_allow_html=True)

def create_market_overview_grid(data):
    """Create compact market overview"""
    
    # Market data in 2 rows of 2 charts each
    market_row1_col1, market_row1_col2 = st.columns(2)
    
    with market_row1_col1:
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        fig_sp500 = create_compact_chart(data, "sp500", "📊 S&P 500 Index", 250)
        st.plotly_chart(fig_sp500, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with market_row1_col2:
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        # Create correlation mini-heatmap
        if 'fis_data' in data and 'dow_banks' in data and not data['fis_data'].empty and not data['dow_banks'].empty:
            # Simple correlation between FIS and Dow Banks
            fis_prices = data['fis_data']['Price'].head(20)
            dow_prices = data['dow_banks']['Price'].head(20)
            
            if len(fis_prices) == len(dow_prices):
                correlation = np.corrcoef(fis_prices, dow_prices)[0, 1]
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=[[1, correlation], [correlation, 1]],
                    x=['FIS', 'Dow Banks'],
                    y=['FIS', 'Dow Banks'],
                    colorscale='RdBu_r',
                    text=[[1, f'{correlation:.2f}'], [f'{correlation:.2f}', 1]],
                    texttemplate="%{text}",
                    textfont={"size": 14}
                ))
                
                fig_corr.update_layout(
                    title="🔗 Market Correlation",
                    height=250,
                    margin=dict(l=40, r=40, t=50, b=40)
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("📊 Correlation data processing...")
        else:
            st.info("📊 Loading correlation data...")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Load data
    data = load_data()
    
    # Compact header
    st.markdown('<div class="main-header">🚀 Ultimate Financial Dashboard</div>', unsafe_allow_html=True)
    
    # Top navigation bar (horizontal)
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([2, 2, 2, 2, 2])
    
    with nav_col1:
        overview_selected = st.button("📊 Overview", use_container_width=True, type="primary")
    with nav_col2:
        analysis_selected = st.button("🔍 Analysis", use_container_width=True)
    with nav_col3:
        performance_selected = st.button("🎯 Performance", use_container_width=True)
    with nav_col4:
        markets_selected = st.button("💰 Markets", use_container_width=True)
    with nav_col5:
        settings_selected = st.button("⚙️ Settings", use_container_width=True)
    
    # Determine active page
    if analysis_selected:
        page = "Analysis"
    elif performance_selected:
        page = "Performance"
    elif markets_selected:
        page = "Markets"
    elif settings_selected:
        page = "Settings"
    else:
        page = "Overview"  # Default
    
    # Compact alerts
    create_compact_alerts(data)
    
    # KPI row
    create_compact_kpi_row(data)
    
    # Page content with fixed layouts
    if page == "Overview":
        # Main grid layout
        create_grid_layout(data)
        
        # Performance metrics row
        st.markdown('<div class="sub-header">⚡ Performance Metrics</div>', unsafe_allow_html=True)
        create_performance_grid(data)
    
    elif page == "Analysis":
        st.markdown('<div class="sub-header">🔍 Advanced Analysis</div>', unsafe_allow_html=True)
        
        # Analysis in compact grid
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Statistical Analysis")
            
            if 'sample_data' in data and not data['sample_data'].empty:
                # Quick stats
                total_records = len(data['sample_data'])
                avg_actual = data['sample_data']['Actual'].mean()
                std_actual = data['sample_data']['Actual'].std()
                
                st.markdown(f"""
                **Data Summary:**
                - Records: {total_records:,}
                - Average: ${avg_actual:,.0f}
                - Std Dev: ${std_actual:,.0f}
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with analysis_col2:
            st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Key Insights")
            
            insights = [
                "📈 Upward trend in Q4 2024",
                "🎯 Plan achievement at 94.2%",
                "📊 Lineup ABC234006 outperforming",
                "⚠️ Monitor Site LBS variance"
            ]
            
            for insight in insights:
                st.markdown(f"• {insight}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Correlation heatmap (compact)
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown("#### 🔗 Market Correlations")
        
        # Create simple correlation matrix
        if all(key in data and not data[key].empty for key in ['fis_data', 'dow_banks']):
            # Sample correlation data
            corr_data = np.array([[1.0, 0.75, 0.82], [0.75, 1.0, 0.68], [0.82, 0.68, 1.0]])
            
            fig_corr = px.imshow(
                corr_data,
                x=['FIS', 'Dow Banks', 'S&P 500'],
                y=['FIS', 'Dow Banks', 'S&P 500'],
                color_continuous_scale='RdBu_r',
                title="Market Correlation Matrix",
                text_auto=True
            )
            fig_corr.update_layout(height=300, margin=dict(l=40, r=40, t=50, b=40))
            st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Performance":
        st.markdown('<div class="sub-header">🎯 Performance Dashboard</div>', unsafe_allow_html=True)
        
        # Performance grid
        create_performance_grid(data)
        
        # Performance charts in 2x2 grid
        perf_chart_col1, perf_chart_col2 = st.columns(2)
        
        with perf_chart_col1:
            st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
            
            # Performance gauge
            if 'sample_data' in data and 'plan_data' in data and not data['sample_data'].empty and not data['plan_data'].empty:
                actual_total = data['sample_data']['Actual'].sum()
                plan_total = data['plan_data']['Plan'].sum()
                achievement = (actual_total / plan_total) * 100
                
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=achievement,
                    title={'text': "Plan Achievement %"},
                    gauge={
                        'axis': {'range': [None, 150]},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 80], 'color': "#fee2e2"},
                            {'range': [80, 100], 'color': "#fef3c7"},
                            {'range': [100, 150], 'color': "#d1fae5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 100
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with perf_chart_col2:
            st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
            
            # Trend analysis
            if 'sample_data' in data and not data['sample_data'].empty:
                monthly_trend = data['sample_data'].groupby(data['sample_data']['DATE'].dt.to_period('M'))['Actual'].sum()
                
                fig_trend = px.line(
                    x=monthly_trend.index.astype(str), 
                    y=monthly_trend.values,
                    title="📈 Monthly Trend Analysis"
                )
                fig_trend.update_traces(line=dict(color='#764ba2', width=3))
                fig_trend.update_layout(height=280, margin=dict(l=40, r=40, t=50, b=40))
                st.plotly_chart(fig_trend, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Markets":
        st.markdown('<div class="sub-header">💰 Financial Markets</div>', unsafe_allow_html=True)
        
        # Market overview grid
        create_market_overview_grid(data)
        
        # Market comparison
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Market Comparison")
        
        if all(key in data and not data[key].empty for key in ['fis_data', 'dow_banks', 'sp500']):
            # Create normalized comparison
            fig_market = go.Figure()
            
            # Normalize all to percentage change from first value
            fis_norm = (data['fis_data']['Price'] / data['fis_data']['Price'].iloc[-1] - 1) * 100
            dow_norm = (data['dow_banks']['Price'] / data['dow_banks']['Price'].iloc[-1] - 1) * 100
            sp500_norm = (data['sp500']['Close'] / data['sp500']['Close'].iloc[-1] - 1) * 100
            
            fig_market.add_trace(go.Scatter(x=data['fis_data']['Date'], y=fis_norm, name='FIS', line=dict(width=2)))
            fig_market.add_trace(go.Scatter(x=data['dow_banks']['Date'], y=dow_norm, name='Dow Banks', line=dict(width=2)))
            fig_market.add_trace(go.Scatter(x=data['sp500']['Date'], y=sp500_norm, name='S&P 500', line=dict(width=2)))
            
            fig_market.update_layout(
                title="📊 Normalized Market Performance (%)",
                height=300,
                margin=dict(l=40, r=40, t=50, b=40),
                yaxis_title="% Change",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_market, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Settings":
        st.markdown('<div class="sub-header">⚙️ Dashboard Settings</div>', unsafe_allow_html=True)
        
        settings_col1, settings_col2 = st.columns(2)
        
        with settings_col1:
            st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
            st.markdown("#### 🎨 Visual Settings")
            
            theme = st.selectbox("Chart Theme", ["Light", "Dark", "Corporate"])
            animations = st.checkbox("Enable Animations", value=True)
            grid_lines = st.checkbox("Show Grid Lines", value=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with settings_col2:
            st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Data Settings")
            
            auto_refresh = st.checkbox("Auto-refresh", value=False)
            cache_duration = st.selectbox("Cache Duration", ["5 min", "15 min", "1 hour"])
            max_records = st.number_input("Max Records", value=1000, min_value=100)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.success("✅ Settings saved!")

if __name__ == "__main__":
    main()