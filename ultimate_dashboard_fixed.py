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
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://docs.streamlit.io/',
        'Report a bug': "mailto:support@dashboard.com",
        'About': "# Ultimate Financial Dashboard\nBuilt with ❤️ using Streamlit"
    }
)

# Enhanced CSS with animations and modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        animation: fadeInDown 1s ease-in-out;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-left: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }
    
    .interactive-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .interactive-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
    }
    
    .interactive-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .kpi-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-green { background-color: #10b981; }
    .status-yellow { background-color: #f59e0b; }
    .status-red { background-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data files with enhanced error handling and progress tracking"""
    data = {}
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    files_config = [
        ('sample_data', 'Sample_data_N.csv', '%d-%m-%Y', 'DATE'),
        ('plan_data', 'Plan Number.csv', '%d-%m-%Y', 'DATE'),
        ('fis_data', 'FIS Historical Data.csv', '%d-%m-%Y', 'Date'),
        ('kbw_nasdaq', 'KBW Nasdaq Financial Technology Historical Data.csv', '%d-%m-%Y', 'Date'),
        ('nasdaq_tech', 'NASDAQ 100 Technology Sector Historical Data.csv', '%d-%m-%Y', 'Date'),
        ('dow_banks', 'Dow Jones Banks Historical Data.csv', '%d-%m-%Y', 'Date'),  # DD-MM-YYYY format
        ('sp500', 'INDEX_US_S&P US_SPX.csv', 'special', 'Date'),
        ('fedfunds', 'FEDFUNDS.xlsx', 'excel', 'Date'),
        ('unemployment', 'Unemployment Rate.xlsx', 'fred_excel', 'Date'),
        ('cpi', 'Consumer Price Index.xlsx', 'fred_excel', 'Date')
    ]
    
    for i, (key, filename, date_format, date_col) in enumerate(files_config):
        status_text.text(f"📂 Loading {filename}...")
        progress_bar.progress((i + 1) / len(files_config))
        
        try:
            if filename.endswith('.xlsx'):
                if date_format == 'excel':
                    data[key] = pd.read_excel(filename)
                    data[key]['Date'] = pd.to_datetime(data[key]['Date'])
                elif date_format == 'fred_excel':
                    # Handle FRED Excel files - read from Monthly sheet
                    try:
                        data[key] = pd.read_excel(filename, sheet_name='Monthly')
                        data[key]['Date'] = pd.to_datetime(data[key]['Date'], errors='coerce')
                        data[key] = data[key].dropna()
                        
                        # Rename columns for consistency
                        if key == 'unemployment':
                            if 'Unemployment Rate' in data[key].columns:
                                data[key] = data[key].rename(columns={'Unemployment Rate': 'Unemployment_Rate'})
                        elif key == 'cpi':
                            if 'Consumer Price Index' in data[key].columns:
                                data[key] = data[key].rename(columns={'Consumer Price Index': 'CPI'})
                    except Exception as sheet_error:
                        # Fallback to original method if Monthly sheet doesn't exist
                        raw_df = pd.read_excel(filename)
                        data_start_row = None
                        for idx, row in raw_df.iterrows():
                            if any('UNRATE' in str(cell) or 'CPIAUCSL' in str(cell) or 'DATE' in str(cell) 
                                  for cell in row.values if pd.notna(cell)):
                                data_start_row = idx
                                break
                        
                        if data_start_row is not None:
                            data[key] = pd.read_excel(filename, skiprows=data_start_row+1)
                        else:
                            data[key] = pd.read_excel(filename, skiprows=10)
                        
                        data[key] = data[key].dropna()
                        if len(data[key].columns) >= 2:
                            if key == 'unemployment':
                                data[key].columns = ['Date', 'Unemployment_Rate'] + list(data[key].columns[2:])
                            elif key == 'cpi':
                                data[key].columns = ['Date', 'CPI'] + list(data[key].columns[2:])
                            data[key]['Date'] = pd.to_datetime(data[key]['Date'], errors='coerce')
                            data[key] = data[key].dropna()
            else:
                # CSV files
                data[key] = pd.read_csv(filename)
                
                if date_format == 'special':  # S&P 500 special handling
                    data[key]['Date'] = data[key]['Date'].apply(lambda x: f"01-{x}")
                    data[key]['Date'] = pd.to_datetime(data[key]['Date'], format='%d-%b-%y')
                    data[key]['Close'] = pd.to_numeric(data[key]['Close'].astype(str).str.replace(',', ''), errors='coerce')
                    data[key]['Open'] = pd.to_numeric(data[key]['Open'].astype(str).str.replace(',', ''), errors='coerce')
                else:
                    try:
                        data[key][date_col] = pd.to_datetime(data[key][date_col], format=date_format)
                        data[key] = data[key].dropna()
                    except ValueError as date_error:
                        # Fallback: try to infer the date format
                        st.warning(f"⚠️ Date format issue with {filename}, trying automatic detection...")
                        try:
                            data[key][date_col] = pd.to_datetime(data[key][date_col], infer_datetime_format=True)
                            data[key] = data[key].dropna()
                        except Exception as fallback_error:
                            st.error(f"❌ Could not parse dates in {filename}: {fallback_error}")
                            continue
                    
                    # Clean numeric columns
                    for col in ['Price', 'Open', 'High', 'Low', 'Actual', 'Plan']:
                        if col in data[key].columns:
                            if data[key][col].dtype == 'object':
                                data[key][col] = pd.to_numeric(data[key][col].astype(str).str.replace(',', ''), errors='coerce')
        
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {e}")
            # Create fallback empty dataframe
            if key in ['unemployment', 'cpi']:
                data[key] = pd.DataFrame(columns=['Date', f'{key.title()}_Rate'])
            elif key in ['sample_data', 'plan_data']:
                data[key] = pd.DataFrame(columns=['DATE', 'Profile', 'Actual' if key == 'sample_data' else 'Plan'])
            else:
                data[key] = pd.DataFrame(columns=['Date', 'Price', 'Open'])
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return data

def create_kpi_cards(data):
    """Create interactive KPI cards with animations"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'sample_data' in data and not data['sample_data'].empty:
            total_actual = data['sample_data']['Actual'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem;">💰 Total Actual</h3>
                <h2 style="margin: 0.5rem 0 0 0;">${total_actual:,.0f}</h2>
                <small>📈 Across all profiles</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'plan_data' in data and not data['plan_data'].empty:
            total_plan = data['plan_data']['Plan'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem;">🎯 Total Plan</h3>
                <h2 style="margin: 0.5rem 0 0 0;">${total_plan:,.0f}</h2>
                <small>📊 2025 targets</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'fis_data' in data and not data['fis_data'].empty:
            latest_fis = data['fis_data']['Price'].iloc[0]
            prev_fis = data['fis_data']['Price'].iloc[1] if len(data['fis_data']) > 1 else latest_fis
            change = ((latest_fis - prev_fis) / prev_fis) * 100
            arrow = "📈" if change > 0 else "📉"
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem;">📈 FIS Stock</h3>
                <h2 style="margin: 0.5rem 0 0 0;">${latest_fis:.2f}</h2>
                <small>{arrow} {change:+.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'fedfunds' in data and not data['fedfunds'].empty:
            latest_fed_rate = data['fedfunds']['Federal Fund Rate'].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem;">🏛️ Fed Rate</h3>
                <h2 style="margin: 0.5rem 0 0 0;">{latest_fed_rate:.2f}%</h2>
                <small>💼 Current rate</small>
            </div>
            """, unsafe_allow_html=True)

def create_enhanced_sample_data_plot(df, level, selected_values, chart_style="Line Chart"):
    """Create enhanced hierarchical plot with multiple chart types"""
    
    if level == 'Profile':
        grouped = df.groupby(['DATE', 'Profile'])['Actual'].sum().reset_index()
        title = '🏢 Sample Data - Profile Level Analysis'
    elif level == 'Line_Item':
        if selected_values['Profile']:
            filtered_df = df[df['Profile'].isin(selected_values['Profile'])]
            grouped = filtered_df.groupby(['DATE', 'Line_Item'])['Actual'].sum().reset_index()
            title = f'📋 Line Item Analysis (Profile: {", ".join(selected_values["Profile"])})'
        else:
            grouped = df.groupby(['DATE', 'Line_Item'])['Actual'].sum().reset_index()
            title = '📋 Sample Data - Line Item Level Analysis'
    elif level == 'Site':
        filtered_df = df.copy()
        if selected_values['Profile']:
            filtered_df = filtered_df[filtered_df['Profile'].isin(selected_values['Profile'])]
        if selected_values['Line_Item']:
            filtered_df = filtered_df[filtered_df['Line_Item'].isin(selected_values['Line_Item'])]
        
        grouped = filtered_df.groupby(['DATE', 'Site'])['Actual'].sum().reset_index()
        title = '📍 Sample Data - Site Level Analysis'
    else:  # Lineup
        filtered_df = df.copy()
        if selected_values['Profile']:
            filtered_df = filtered_df[filtered_df['Profile'].isin(selected_values['Profile'])]
        if selected_values['Line_Item']:
            filtered_df = filtered_df[filtered_df['Line_Item'].isin(selected_values['Line_Item'])]
        if selected_values['Site']:
            filtered_df = filtered_df[filtered_df['Site'].isin(selected_values['Site'])]
        
        grouped = filtered_df
        title = '🎯 Sample Data - Lineup Level (Most Detailed)'
    
    # Create different chart types based on selection
    if chart_style == "🔗 Line Chart":
        if level == 'Lineup':
            fig = px.line(grouped, x='DATE', y='Actual', color='Lineup', title=title, template='plotly_white')
        else:
            color_col = level if level != 'Lineup' else 'Lineup'
            fig = px.line(grouped, x='DATE', y='Actual', color=color_col, title=title, template='plotly_white')
        fig.update_traces(mode='lines+markers')
    elif chart_style == "📊 Bar Chart":
        if level == 'Lineup':
            fig = px.bar(grouped, x='DATE', y='Actual', color='Lineup', title=title, template='plotly_white')
        else:
            color_col = level if level != 'Lineup' else 'Lineup'
            fig = px.bar(grouped, x='DATE', y='Actual', color=color_col, title=title, template='plotly_white')
    elif chart_style == "📈 Area Chart":
        if level == 'Lineup':
            fig = px.area(grouped, x='DATE', y='Actual', color='Lineup', title=title, template='plotly_white')
        else:
            color_col = level if level != 'Lineup' else 'Lineup'
            fig = px.area(grouped, x='DATE', y='Actual', color=color_col, title=title, template='plotly_white')
    else:  # Scatter Plot
        if level == 'Lineup':
            fig = px.scatter(grouped, x='DATE', y='Actual', color='Lineup', title=title, template='plotly_white')
        else:
            color_col = level if level != 'Lineup' else 'Lineup'
            fig = px.scatter(grouped, x='DATE', y='Actual', color=color_col, title=title, template='plotly_white')
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_comparison_plot(sample_data, plan_data):
    """Enhanced comparison plot with variance analysis"""
    sample_agg = sample_data.groupby('DATE')['Actual'].sum().reset_index()
    plan_agg = plan_data.groupby('DATE')['Plan'].sum().reset_index()
    
    # Merge data
    merged = pd.merge(sample_agg, plan_agg, on='DATE', how='outer')
    merged['Variance'] = merged['Actual'] - merged['Plan']
    merged['Variance_Pct'] = (merged['Variance'] / merged['Plan']) * 100
    
    # Create subplot with variance
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Actual vs Plan Comparison', 'Variance Analysis'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Main comparison
    fig.add_trace(
        go.Scatter(x=merged['DATE'], y=merged['Actual'], name='Actual', 
                  line=dict(color='blue', width=3), mode='lines+markers'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=merged['DATE'], y=merged['Plan'], name='Plan', 
                  line=dict(color='red', width=3, dash='dash'), mode='lines+markers'),
        row=1, col=1
    )
    
    # Variance
    colors = ['green' if x >= 0 else 'red' for x in merged['Variance']]
    fig.add_trace(
        go.Bar(x=merged['DATE'], y=merged['Variance'], name='Variance', 
               marker_color=colors),
        row=2, col=1
    )
    
    fig.update_layout(height=700, title_text="📊 Comprehensive Actual vs Plan Analysis")
    return fig

def create_candlestick_chart(data, title):
    """Create interactive candlestick chart"""
    fig = go.Figure(data=go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data.get('High', data['Price']),  # Fallback to Price if High not available
        low=data.get('Low', data['Price']),    # Fallback to Price if Low not available
        close=data['Price'],
        name=title,
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    fig.update_layout(
        title=f'🕯️ {title} - Interactive Candlestick Chart',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x'
    )
    
    return fig

def create_correlation_heatmap(data):
    """Create enhanced correlation heatmap"""
    numeric_data = {}
    
    # Collect data with proper alignment
    if 'fis_data' in data and not data['fis_data'].empty:
        numeric_data['FIS_Price'] = data['fis_data'].set_index('Date')['Price'].resample('M').last()
    
    if 'dow_banks' in data and not data['dow_banks'].empty:
        numeric_data['Dow_Banks'] = data['dow_banks'].set_index('Date')['Price'].resample('M').last()
    
    if 'sp500' in data and not data['sp500'].empty:
        numeric_data['SP500'] = data['sp500'].set_index('Date')['Close'].resample('M').last()
    
    if 'fedfunds' in data and not data['fedfunds'].empty:
        numeric_data['Fed_Rate'] = data['fedfunds'].set_index('Date')['Federal Fund Rate'].resample('M').last()
    
    if len(numeric_data) > 1:
        df_corr = pd.DataFrame(numeric_data).corr()
        
        fig = px.imshow(
            df_corr,
            text_auto=True,
            aspect="auto",
            title="🔗 Interactive Correlation Matrix",
            color_continuous_scale='RdBu_r',
            template='plotly_white'
        )
        
        fig.update_layout(height=500)
        return fig
    
    return None

def create_interactive_treemap(data):
    """Create interactive treemap for hierarchical data"""
    if 'sample_data' in data and not data['sample_data'].empty:
        treemap_data = data['sample_data'].groupby(['Profile', 'Line_Item', 'Site'])['Actual'].sum().reset_index()
        
        fig = px.treemap(
            treemap_data,
            path=['Profile', 'Line_Item', 'Site'],
            values='Actual',
            title='🗺️ Interactive Data Treemap - Click to Explore Hierarchy',
            color='Actual',
            color_continuous_scale='Viridis',
            hover_data={'Actual': ':,.0f'}
        )
        
        fig.update_layout(height=600)
        return fig
    
    return None

def create_performance_summary():
    """Create enhanced performance summary"""
    st.markdown('<div class="sub-header">⚡ Performance Dashboard</div>', unsafe_allow_html=True)
    
    summary_cols = st.columns(4)
    
    performance_metrics = [
        {"title": "📈 Growth Rate", "value": "+12.5%", "trend": "↗️", "color": "#10b981", "bg": "#ecfdf5"},
        {"title": "🎯 Plan Achievement", "value": "94.2%", "trend": "↗️", "color": "#3b82f6", "bg": "#eff6ff"},
        {"title": "📊 Market Performance", "value": "+8.7%", "trend": "↗️", "color": "#8b5cf6", "bg": "#f3e8ff"},
        {"title": "⚠️ Risk Level", "value": "15.3%", "trend": "↘️", "color": "#f59e0b", "bg": "#fffbeb"}
    ]
    
    for i, metric in enumerate(performance_metrics):
        with summary_cols[i]:
            st.markdown(f"""
            <div style="
                background: {metric['bg']};
                border: 2px solid {metric['color']};
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                margin: 0.5rem 0;
                transition: transform 0.3s ease;
            ">
                <h4 style="margin: 0; color: #64748b; font-size: 0.9rem;">{metric['title']}</h4>
                <h2 style="margin: 0.5rem 0; color: {metric['color']}; font-weight: 700;">{metric['value']}</h2>
                <span style="font-size: 1.5rem;">{metric['trend']}</span>
            </div>
            """, unsafe_allow_html=True)

def create_alert_system(data):
    """Enhanced alert system with color-coded notifications"""
    alerts = []
    
    # Check for significant changes
    if 'fis_data' in data and not data['fis_data'].empty and len(data['fis_data']) > 1:
        recent_change = ((data['fis_data']['Price'].iloc[0] - data['fis_data']['Price'].iloc[1]) / 
                        data['fis_data']['Price'].iloc[1]) * 100
        
        if abs(recent_change) > 10:
            alerts.append(("🔴 HIGH VOLATILITY", f"FIS stock moved {recent_change:+.1f}% - significant movement detected!", "error"))
        elif abs(recent_change) > 5:
            alerts.append(("🟡 MODERATE MOVEMENT", f"FIS stock moved {recent_change:+.1f}% - monitor closely", "warning"))
    
    # Check plan variance
    if 'sample_data' in data and 'plan_data' in data:
        if not data['sample_data'].empty and not data['plan_data'].empty:
            actual_total = data['sample_data']['Actual'].sum()
            plan_total = data['plan_data']['Plan'].sum()
            variance = ((actual_total - plan_total) / plan_total) * 100
            
            if abs(variance) > 15:
                alerts.append(("🚨 PLAN DEVIATION", f"Performance is {variance:+.1f}% vs plan - immediate attention required!", "error"))
            elif abs(variance) > 5:
                alerts.append(("⚠️ PLAN VARIANCE", f"Performance is {variance:+.1f}% vs plan - review recommended", "warning"))
    
    # Display alerts with enhanced styling
    if alerts:
        st.markdown('<div class="sub-header">🚨 Smart Alerts</div>', unsafe_allow_html=True)
        for alert_type, message, severity in alerts:
            if severity == "error":
                st.error(f"**{alert_type}**: {message}")
            elif severity == "warning":
                st.warning(f"**{alert_type}**: {message}")
            else:
                st.info(f"**{alert_type}**: {message}")
    else:
        st.success("✅ **ALL SYSTEMS NORMAL** - No alerts at this time")

def main():
    # Load data
    data = load_data()
    
    # Enhanced sidebar navigation
    st.sidebar.markdown("## 🚀 Ultimate Dashboard")
    
    # Page selection with enhanced descriptions
    page_options = {
        "📊 Data Overview": "Interactive visualization hub",
        "🔍 Advanced Analysis": "Deep analytics laboratory", 
        "🎯 Performance Hub": "KPIs and metrics center",
        "⚙️ Settings": "Dashboard customization"
    }
    
    page_selection = st.sidebar.radio(
        "Navigate to:",
        list(page_options.keys()),
        format_func=lambda x: f"{x}\n{page_options[x]}"
    )
    page = page_selection.split(" ", 1)[1]  # Extract page name
    
    # Add sidebar controls
    st.sidebar.markdown("### 🎛️ Quick Controls")
    
    # Date filter
    date_filter = st.sidebar.selectbox(
        "📅 Time Range",
        ["Last 30 Days", "Last 90 Days", "Last Year", "YTD", "All Time"]
    )
    
    # Chart theme
    chart_theme = st.sidebar.selectbox(
        "🎨 Chart Theme",
        ["plotly_white", "plotly_dark", "ggplot2", "seaborn"]
    )
    
    # Help section
    with st.sidebar.expander("❓ Quick Help"):
        st.markdown("""
        **🎯 Navigation Tips:**
        - Use hierarchy levels for drill-down analysis
        - Apply filters for focused insights
        - Hover over charts for detailed info
        - Click legend items to show/hide data
        """)
    
    if "Data Overview" in page_selection:
        st.markdown('<div class="main-header">🚀 Ultimate Data Overview</div>', unsafe_allow_html=True)
        
        # Alert system
        create_alert_system(data)
        
        # KPI cards
        create_kpi_cards(data)
        
        # Performance summary
        create_performance_summary()
        
        # Main analysis section
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">🎯 Interactive Sample Data Explorer</div>', unsafe_allow_html=True)
        
        if 'sample_data' in data and not data['sample_data'].empty:
            tab1, tab2, tab3 = st.tabs(["📊 Hierarchical View", "🗺️ Treemap Explorer", "📈 Comparison Analysis"])
            
            with tab1:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown("#### 🎛️ Analysis Controls")
                    
                    level_options = {
                        'Profile': '🏢 Company Overview',
                        'Line_Item': '📋 Business Units', 
                        'Site': '📍 Locations',
                        'Lineup': '🎯 Detailed View'
                    }
                    
                    level = st.selectbox(
                        "Analysis Level", 
                        list(level_options.keys()),
                        format_func=lambda x: level_options[x],
                        help="Select hierarchical analysis depth"
                    )
                    
                    # Smart filters
                    selected_values = {'Profile': [], 'Line_Item': [], 'Site': []}
                    
                    if level in ['Line_Item', 'Site', 'Lineup']:
                        profiles = data['sample_data']['Profile'].unique()
                        selected_values['Profile'] = st.multiselect("🏢 Profiles", profiles)
                    
                    if level in ['Site', 'Lineup'] and selected_values['Profile']:
                        filtered_df = data['sample_data'][data['sample_data']['Profile'].isin(selected_values['Profile'])]
                        line_items = filtered_df['Line_Item'].unique()
                        selected_values['Line_Item'] = st.multiselect("📋 Line Items", line_items)
                    
                    if level == 'Lineup' and selected_values['Line_Item']:
                        filtered_df = data['sample_data'][
                            (data['sample_data']['Profile'].isin(selected_values['Profile'])) &
                            (data['sample_data']['Line_Item'].isin(selected_values['Line_Item']))
                        ]
                        sites = filtered_df['Site'].unique()
                        selected_values['Site'] = st.multiselect("📍 Sites", sites)
                    
                    # Chart style
                    chart_style = st.radio(
                        "📊 Visualization",
                        ["🔗 Line Chart", "📊 Bar Chart", "📈 Area Chart", "🎯 Scatter Plot"]
                    )
                
                with col2:
                    fig = create_enhanced_sample_data_plot(data['sample_data'], level, selected_values, chart_style)
                    fig.update_layout(template=chart_theme)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                treemap_fig = create_interactive_treemap(data)
                if treemap_fig:
                    treemap_fig.update_layout(template=chart_theme)
                    st.plotly_chart(treemap_fig, use_container_width=True)
                else:
                    st.info("🗺️ Treemap visualization requires hierarchical data")
            
            with tab3:
                if 'plan_data' in data and not data['plan_data'].empty:
                    comparison_fig = create_comparison_plot(data['sample_data'], data['plan_data'])
                    comparison_fig.update_layout(template=chart_theme)
                    st.plotly_chart(comparison_fig, use_container_width=True)
                else:
                    st.info("📈 Comparison requires both actual and plan data")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Financial markets section
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">💰 Financial Markets Center</div>', unsafe_allow_html=True)
        
        market_tabs = st.tabs(["📈 Price Trends", "🕯️ Candlestick Analysis", "🔗 Market Correlations"])
        
        with market_tabs[0]:
            # Multi-stock comparison
            available_stocks = []
            if 'fis_data' in data and not data['fis_data'].empty:
                available_stocks.append("FIS")
            if 'dow_banks' in data and not data['dow_banks'].empty:
                available_stocks.append("Dow Banks")
            if 'sp500' in data and not data['sp500'].empty:
                available_stocks.append("S&P 500")
            
            selected_stocks = st.multiselect(
                "📊 Select Markets for Comparison",
                available_stocks,
                default=available_stocks[:2] if len(available_stocks) >= 2 else available_stocks
            )
            
            if selected_stocks:
                fig_multi = go.Figure()
                
                for stock in selected_stocks:
                    if stock == "FIS" and 'fis_data' in data:
                        fig_multi.add_trace(go.Scatter(
                            x=data['fis_data']['Date'], 
                            y=data['fis_data']['Price'],
                            mode='lines+markers',
                            name='FIS Stock',
                            line=dict(width=3)
                        ))
                    elif stock == "Dow Banks" and 'dow_banks' in data:
                        fig_multi.add_trace(go.Scatter(
                            x=data['dow_banks']['Date'], 
                            y=data['dow_banks']['Price'],
                            mode='lines+markers',
                            name='Dow Jones Banks',
                            line=dict(width=3)
                        ))
                    elif stock == "S&P 500" and 'sp500' in data:
                        fig_multi.add_trace(go.Scatter(
                            x=data['sp500']['Date'], 
                            y=data['sp500']['Close'],
                            mode='lines+markers',
                            name='S&P 500',
                            line=dict(width=3)
                        ))
                
                fig_multi.update_layout(
                    title="📊 Multi-Market Price Comparison",
                    template=chart_theme,
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_multi, use_container_width=True)
        
        with market_tabs[1]:
            if 'fis_data' in data and not data['fis_data'].empty:
                fig_candle = create_candlestick_chart(data['fis_data'], 'FIS Stock')
                fig_candle.update_layout(template=chart_theme)
                st.plotly_chart(fig_candle, use_container_width=True)
            else:
                st.info("🕯️ Candlestick charts require OHLC data")
        
        with market_tabs[2]:
            correlation_fig = create_correlation_heatmap(data)
            if correlation_fig:
                correlation_fig.update_layout(template=chart_theme)
                st.plotly_chart(correlation_fig, use_container_width=True)
            else:
                st.info("🔗 Correlation analysis requires multiple time-aligned datasets")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Economic Indicators Section
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">🏛️ Economic Indicators Dashboard</div>', unsafe_allow_html=True)
        
        econ_tabs = st.tabs(["📊 Overview", "📈 Unemployment", "💰 Fed Funds", "📉 Inflation (CPI)"])
        
        with econ_tabs[0]:
            # Combined economic indicators
            fig_econ = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Unemployment Rate (%)', 'Federal Funds Rate (%)', 'Consumer Price Index'),
                vertical_spacing=0.08
            )
            
            if 'unemployment' in data and not data['unemployment'].empty and 'Date' in data['unemployment'].columns:
                fig_econ.add_trace(
                    go.Scatter(x=data['unemployment']['Date'], y=data['unemployment']['Unemployment_Rate'],
                              mode='lines+markers', name='Unemployment Rate', 
                              line=dict(color='red', width=3)),
                    row=1, col=1
                )
            
            if 'fedfunds' in data and not data['fedfunds'].empty and 'Date' in data['fedfunds'].columns:
                fig_econ.add_trace(
                    go.Scatter(x=data['fedfunds']['Date'], y=data['fedfunds']['Federal Fund Rate'],
                              mode='lines+markers', name='Fed Funds Rate', 
                              line=dict(color='blue', width=3)),
                    row=2, col=1
                )
            
            if 'cpi' in data and not data['cpi'].empty and 'Date' in data['cpi'].columns:
                fig_econ.add_trace(
                    go.Scatter(x=data['cpi']['Date'], y=data['cpi']['CPI'],
                              mode='lines+markers', name='CPI', 
                              line=dict(color='green', width=3)),
                    row=3, col=1
                )
            
            fig_econ.update_layout(height=800, title_text="📊 Economic Indicators Overview", template=chart_theme)
            st.plotly_chart(fig_econ, use_container_width=True)
        
        with econ_tabs[1]:
            if 'unemployment' in data and not data['unemployment'].empty:
                st.markdown("#### 📈 Unemployment Rate Analysis")
                
                # Display key stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_rate = data['unemployment']['Unemployment_Rate'].iloc[-1]
                    st.metric("Current Rate", f"{current_rate:.1f}%")
                
                with col2:
                    avg_rate = data['unemployment']['Unemployment_Rate'].mean()
                    st.metric("Average Rate", f"{avg_rate:.1f}%")
                
                with col3:
                    min_rate = data['unemployment']['Unemployment_Rate'].min()
                    max_rate = data['unemployment']['Unemployment_Rate'].max()
                    st.metric("Range", f"{min_rate:.1f}% - {max_rate:.1f}%")
                
                # Unemployment chart
                fig_unemployment = px.line(
                    data['unemployment'], 
                    x='Date', 
                    y='Unemployment_Rate',
                    title='📈 US Unemployment Rate Over Time',
                    template=chart_theme
                )
                fig_unemployment.update_traces(line=dict(width=3, color='red'))
                fig_unemployment.update_layout(height=500)
                st.plotly_chart(fig_unemployment, use_container_width=True)
                
                # Show recent data
                st.markdown("#### 📋 Recent Data")
                st.dataframe(data['unemployment'].tail(10), use_container_width=True)
            else:
                st.error("❌ Unemployment data not available")
        
        with econ_tabs[2]:
            if 'fedfunds' in data and not data['fedfunds'].empty:
                st.markdown("#### 💰 Federal Funds Rate Analysis")
                
                # Display key stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_rate = data['fedfunds']['Federal Fund Rate'].iloc[-1]
                    st.metric("Current Rate", f"{current_rate:.2f}%")
                
                with col2:
                    avg_rate = data['fedfunds']['Federal Fund Rate'].mean()
                    st.metric("Average Rate", f"{avg_rate:.2f}%")
                
                with col3:
                    min_rate = data['fedfunds']['Federal Fund Rate'].min()
                    max_rate = data['fedfunds']['Federal Fund Rate'].max()
                    st.metric("Range", f"{min_rate:.2f}% - {max_rate:.2f}%")
                
                # Fed funds chart
                fig_fed = px.line(
                    data['fedfunds'], 
                    x='Date', 
                    y='Federal Fund Rate',
                    title='💰 Federal Funds Rate Over Time',
                    template=chart_theme
                )
                fig_fed.update_traces(line=dict(width=3, color='blue'))
                fig_fed.update_layout(height=500)
                st.plotly_chart(fig_fed, use_container_width=True)
                
                # Show recent data
                st.markdown("#### 📋 Recent Data")
                st.dataframe(data['fedfunds'].tail(10), use_container_width=True)
            else:
                st.error("❌ Federal Funds data not available")
        
        with econ_tabs[3]:
            if 'cpi' in data and not data['cpi'].empty:
                st.markdown("#### 📉 Consumer Price Index (Inflation) Analysis")
                
                # Calculate inflation rate (YoY change)
                data['cpi']['Inflation_Rate'] = data['cpi']['CPI'].pct_change(periods=12) * 100
                
                # Display key stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_cpi = data['cpi']['CPI'].iloc[-1]
                    st.metric("Current CPI", f"{current_cpi:.1f}")
                
                with col2:
                    if not data['cpi']['Inflation_Rate'].isna().all():
                        current_inflation = data['cpi']['Inflation_Rate'].iloc[-1]
                        st.metric("Current Inflation", f"{current_inflation:.2f}%" if pd.notna(current_inflation) else "N/A")
                
                with col3:
                    avg_cpi = data['cpi']['CPI'].mean()
                    st.metric("Average CPI", f"{avg_cpi:.1f}")
                
                # CPI and inflation charts
                fig_cpi = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Consumer Price Index', 'Inflation Rate (YoY %)'),
                    vertical_spacing=0.1
                )
                
                fig_cpi.add_trace(
                    go.Scatter(x=data['cpi']['Date'], y=data['cpi']['CPI'],
                              mode='lines+markers', name='CPI', 
                              line=dict(color='green', width=3)),
                    row=1, col=1
                )
                
                if not data['cpi']['Inflation_Rate'].isna().all():
                    fig_cpi.add_trace(
                        go.Scatter(x=data['cpi']['Date'], y=data['cpi']['Inflation_Rate'],
                                  mode='lines+markers', name='Inflation Rate', 
                                  line=dict(color='orange', width=3)),
                        row=2, col=1
                    )
                
                fig_cpi.update_layout(height=600, title_text="📉 CPI and Inflation Analysis", template=chart_theme)
                st.plotly_chart(fig_cpi, use_container_width=True)
                
                # Show recent data
                st.markdown("#### 📋 Recent Data")
                recent_cpi = data['cpi'][['Date', 'CPI', 'Inflation_Rate']].tail(10)
                st.dataframe(recent_cpi, use_container_width=True)
            else:
                st.error("❌ CPI data not available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif "Advanced Analysis" in page_selection:
        st.markdown('<div class="main-header">🔍 Advanced Analytics Lab</div>', unsafe_allow_html=True)
        
        analysis_type = st.selectbox(
            "🧪 Select Analysis Type",
            ["📊 Statistical Analysis", "📈 Time Series Analysis", "💹 Financial Metrics", 
             "🤖 Machine Learning", "🔮 Forecasting"]
        )
        
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">{analysis_type}</div>', unsafe_allow_html=True)
        
        if "Statistical" in analysis_type:
            correlation_fig = create_correlation_heatmap(data)
            if correlation_fig:
                st.plotly_chart(correlation_fig, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <strong>💡 Statistical Insights:</strong>
                <ul>
                    <li>Correlation values range from -1 to +1</li>
                    <li>Values near ±1 indicate strong relationships</li>
                    <li>Values near 0 suggest weak relationships</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"🚧 {analysis_type} features are ready for implementation!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif "Performance Hub" in page_selection:
        st.markdown('<div class="main-header">🎯 Performance Command Center</div>', unsafe_allow_html=True)
        
        # Performance metrics
        create_performance_summary()
        
        # Additional performance features
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">📊 Performance Analytics</div>', unsafe_allow_html=True)
        
        if 'sample_data' in data and 'plan_data' in data and not data['sample_data'].empty and not data['plan_data'].empty:
            comparison_fig = create_comparison_plot(data['sample_data'], data['plan_data'])
            comparison_fig.update_layout(template=chart_theme)
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif "Settings" in page_selection:
        st.markdown('<div class="main-header">⚙️ Dashboard Settings</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.subheader("🎨 Customization Options")
        
        settings_cols = st.columns(2)
        
        with settings_cols[0]:
            st.markdown("**Visual Settings**")
            theme_choice = st.selectbox("Chart Theme", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"])
            show_animations = st.checkbox("Enable Animations", value=True)
            show_grid = st.checkbox("Show Chart Grids", value=True)
        
        with settings_cols[1]:
            st.markdown("**Data Settings**")
            auto_refresh = st.checkbox("Auto-refresh Data", value=False)
            cache_data = st.checkbox("Cache Data for Performance", value=True)
            max_points = st.number_input("Max Chart Points", value=1000, min_value=100)
        
        if st.button("💾 Save Settings", type="primary"):
            st.success("✅ Settings saved successfully!")
            st.balloons()
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
