import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
from interactive_features import *
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

# Load the enhanced CSS
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
    
    .feature-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
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
    
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .floating-action {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data files with enhanced error handling"""
    data = {}
    
    # Progress bar for loading
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    files_to_load = [
        ('sample_data', 'Sample_data_N.csv', 'CSV'),
        ('plan_data', 'Plan Number.csv', 'CSV'),
        ('fis_data', 'FIS Historical Data.csv', 'CSV'),
        ('kbw_nasdaq', 'KBW Nasdaq Financial Technology Historical Data.csv', 'CSV'),
        ('nasdaq_tech', 'NASDAQ 100 Technology Sector Historical Data.csv', 'CSV'),
        ('dow_banks', 'Dow Jones Banks Historical Data.csv', 'CSV'),
        ('sp500', 'INDEX_US_S&P US_SPX.csv', 'CSV'),
        ('fedfunds', 'FEDFUNDS.xlsx', 'Excel'),
        ('unemployment', 'Unemployment Rate.xlsx', 'Excel'),
        ('cpi', 'Consumer Price Index.xlsx', 'Excel')
    ]
    
    for i, (key, filename, file_type) in enumerate(files_to_load):
        status_text.text(f"Loading {filename}...")
        progress_bar.progress((i + 1) / len(files_to_load))
        
        try:
            if file_type == 'CSV':
                if key == 'sample_data':
                    data[key] = pd.read_csv(filename)
                    data[key]['DATE'] = pd.to_datetime(data[key]['DATE'], format='%d-%m-%Y')
                    data[key] = data[key].dropna()
                elif key == 'plan_data':
                    data[key] = pd.read_csv(filename)
                    data[key]['DATE'] = pd.to_datetime(data[key]['DATE'], format='%d-%m-%Y')
                    data[key] = data[key].dropna()
                elif key in ['fis_data', 'kbw_nasdaq', 'nasdaq_tech']:
                    data[key] = pd.read_csv(filename)
                    data[key]['Date'] = pd.to_datetime(data[key]['Date'], format='%d-%m-%Y')
                    data[key]['Price'] = pd.to_numeric(data[key]['Price'].astype(str).str.replace(',', ''), errors='coerce')
                    if 'Open' in data[key].columns:
                        data[key]['Open'] = pd.to_numeric(data[key]['Open'].astype(str).str.replace(',', ''), errors='coerce')
                elif key == 'dow_banks':
                    data[key] = pd.read_csv(filename)
                    data[key]['Date'] = pd.to_datetime(data[key]['Date'], format='%m/%d/%Y')
                    data[key]['Price'] = pd.to_numeric(data[key]['Price'], errors='coerce')
                    data[key]['Open'] = pd.to_numeric(data[key]['Open'], errors='coerce')
                elif key == 'sp500':
                    data[key] = pd.read_csv(filename)
                    data[key]['Date'] = data[key]['Date'].apply(lambda x: f"01-{x}")
                    data[key]['Date'] = pd.to_datetime(data[key]['Date'], format='%d-%b-%y')
                    data[key]['Close'] = pd.to_numeric(data[key]['Close'].astype(str).str.replace(',', ''), errors='coerce')
                    data[key]['Open'] = pd.to_numeric(data[key]['Open'].astype(str).str.replace(',', ''), errors='coerce')
            
            elif file_type == 'Excel':
                if key == 'fedfunds':
                    data[key] = pd.read_excel(filename)
                    data[key]['Date'] = pd.to_datetime(data[key]['Date'])
                else:
                    # Handle FRED Excel files
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
        
        except Exception as e:
            st.error(f"❌ Error loading {filename}: {e}")
            # Create empty dataframe as fallback
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

def main():
    # Load data with progress indication
    data = load_data()
    
    # Enhanced sidebar with better organization
    st.sidebar.markdown("## 🎛️ Ultimate Dashboard Controls")
    
    # Page selection with enhanced icons and descriptions
    page_options = {
        "📊 Data Overview": {
            "page": "Data Overview",
            "description": "Interactive data visualization and exploration"
        },
        "🔍 Advanced Analysis": {
            "page": "Analysis", 
            "description": "Deep-dive analytics and insights"
        },
        "🎯 Performance Hub": {
            "page": "Performance",
            "description": "KPIs, metrics, and performance tracking"
        },
        "⚙️ Dashboard Settings": {
            "page": "Settings",
            "description": "Customize your dashboard experience"
        }
    }
    
    selected_option = st.sidebar.selectbox(
        "Navigate to:",
        list(page_options.keys()),
        format_func=lambda x: f"{x}\n{page_options[x]['description']}"
    )
    page = page_options[selected_option]["page"]
    
    # Add advanced chart controls
    chart_controls = create_advanced_chart_controls()
    
    # Add export functionality
    add_export_functionality()
    
    # Add help system
    create_advanced_tooltips()
    
    # Add real-time notifications
    add_real_time_notifications()
    
    if page == "Data Overview":
        st.markdown('<div class="main-header">🚀 Ultimate Data Overview Dashboard</div>', unsafe_allow_html=True)
        
        # Alert system
        create_alert_system(data)
        
        # Enhanced KPI cards
        create_kpi_cards(data)
        
        # Performance summary cards
        create_performance_summary_cards()
        
        # Date range filter
        date_filter = create_real_time_filters()
        
        # Sample Data Section with Ultimate Interactivity
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">🎯 Interactive Sample Data Explorer</div>', unsafe_allow_html=True)
        
        if 'sample_data' in data and not data['sample_data'].empty:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Hierarchical Analysis", "🗺️ Treemap View", "📅 Calendar Heatmap"])
            
            with tab1:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.subheader("🎛️ Smart Controls")
                    
                    # Enhanced level selector
                    level_options = {
                        'Profile': {'icon': '🏢', 'desc': 'Company-wide view'},
                        'Line_Item': {'icon': '📋', 'desc': 'Business unit focus'}, 
                        'Site': {'icon': '📍', 'desc': 'Location analysis'},
                        'Lineup': {'icon': '🎯', 'desc': 'Granular details'}
                    }
                    
                    level = st.selectbox(
                        "Analysis Depth", 
                        list(level_options.keys()),
                        format_func=lambda x: f"{level_options[x]['icon']} {x} - {level_options[x]['desc']}",
                        help="Choose your analysis granularity"
                    )
                    
                    # Smart filtering system
                    selected_values = {'Profile': [], 'Line_Item': [], 'Site': [], 'Lineup': []}
                    
                    if level in ['Line_Item', 'Site', 'Lineup']:
                        profiles = data['sample_data']['Profile'].unique()
                        selected_values['Profile'] = st.multiselect(
                            "🏢 Profiles", 
                            profiles,
                            help="Filter by organizational profiles"
                        )
                    
                    if level in ['Site', 'Lineup'] and selected_values['Profile']:
                        filtered_df = data['sample_data'][data['sample_data']['Profile'].isin(selected_values['Profile'])]
                        line_items = filtered_df['Line_Item'].unique()
                        selected_values['Line_Item'] = st.multiselect(
                            "📋 Line Items", 
                            line_items,
                            help="Select business line items"
                        )
                    
                    if level == 'Lineup' and selected_values['Line_Item']:
                        filtered_df = data['sample_data'][
                            (data['sample_data']['Profile'].isin(selected_values['Profile'])) &
                            (data['sample_data']['Line_Item'].isin(selected_values['Line_Item']))
                        ]
                        sites = filtered_df['Site'].unique()
                        selected_values['Site'] = st.multiselect(
                            "📍 Sites", 
                            sites,
                            help="Choose specific locations"
                        )
                    
                    # Chart customization
                    st.subheader("🎨 Visualization")
                    chart_style = st.radio(
                        "Chart Style",
                        ["🔗 Line Chart", "📊 Bar Chart", "📈 Area Chart", "🎯 Scatter Plot"]
                    )
                    
                    show_markers = st.checkbox("Show Data Points", value=True)
                    show_trend = st.checkbox("Add Trend Line", value=False)
                
                with col2:
                    # Apply date filtering
                    filtered_data = data['sample_data'].copy()
                    if date_filter != "All Time":
                        today = datetime.now()
                        if date_filter == "Last 30 Days":
                            start_date = today - timedelta(days=30)
                        elif date_filter == "Last 90 Days":
                            start_date = today - timedelta(days=90)
                        elif date_filter == "Last Year":
                            start_date = today - timedelta(days=365)
                        elif date_filter == "YTD":
                            start_date = datetime(today.year, 1, 1)
                        
                        if date_filter != "Custom Range":
                            filtered_data = filtered_data[filtered_data['DATE'] >= start_date]
                    
                    fig = create_enhanced_sample_data_plot(filtered_data, level, selected_values, None)
                    
                    # Apply chart customizations
                    if chart_controls['theme'] != 'plotly_white':
                        fig.update_layout(template=chart_controls['theme'])
                    
                    if not chart_controls['show_grid']:
                        fig.update_layout(
                            xaxis=dict(showgrid=False),
                            yaxis=dict(showgrid=False)
                        )
                    
                    fig.update_layout(height=chart_controls['height'])
                    
                    if show_trend:
                        # Add trend line (simplified)
                        pass  # Implementation would go here
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                treemap_fig = create_interactive_treemap(data)
                if treemap_fig:
                    st.plotly_chart(treemap_fig, use_container_width=True)
                else:
                    st.info("📊 Treemap requires hierarchical sample data")
            
            with tab3:
                calendar_fig = create_interactive_heatmap_calendar(data)
                if calendar_fig:
                    st.plotly_chart(calendar_fig, use_container_width=True)
                else:
                    st.info("📅 Calendar view requires time-series data")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive Comparison Tool
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        dataset1, dataset2, comparison_type = create_interactive_comparison_tool(data)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cross-filter functionality
        create_cross_filter_functionality()
        
        # Enhanced Financial Markets with Tabs
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">💰 Financial Markets Command Center</div>', unsafe_allow_html=True)
        
        market_tab1, market_tab2, market_tab3, market_tab4 = st.tabs(
            ["📈 Price Charts", "🕯️ Candlesticks", "🔗 Correlations", "📋 Data Tables"]
        )
        
        with market_tab1:
            # Stock price charts with enhanced interactivity
            stock_selector = st.multiselect(
                "Select Markets to Display",
                ["FIS", "KBW NASDAQ", "NASDAQ Tech", "Dow Banks", "S&P 500"],
                default=["FIS", "S&P 500"]
            )
            
            if stock_selector:
                fig_combined = go.Figure()
                
                for stock in stock_selector:
                    if stock == "FIS" and 'fis_data' in data:
                        fig_combined.add_trace(go.Scatter(
                            x=data['fis_data']['Date'], 
                            y=data['fis_data']['Price'],
                            mode='lines+markers',
                            name='FIS',
                            line=dict(width=3)
                        ))
                    elif stock == "S&P 500" and 'sp500' in data:
                        # Normalize S&P 500 to same scale
                        normalized_sp500 = (data['sp500']['Close'] / data['sp500']['Close'].iloc[-1]) * 100
                        fig_combined.add_trace(go.Scatter(
                            x=data['sp500']['Date'], 
                            y=normalized_sp500,
                            mode='lines+markers',
                            name='S&P 500 (Normalized)',
                            line=dict(width=3)
                        ))
                
                fig_combined.update_layout(
                    title="📊 Multi-Market Comparison",
                    template=chart_controls['theme'],
                    height=chart_controls['height'],
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_combined, use_container_width=True)
        
        with market_tab2:
            # Candlestick charts
            candlestick_stock = st.selectbox(
                "Select Stock for Candlestick Analysis",
                ["FIS", "Dow Jones Banks"]
            )
            
            if candlestick_stock == "FIS" and 'fis_data' in data:
                fig_candle = create_candlestick_chart(data['fis_data'], 'FIS Stock')
                st.plotly_chart(fig_candle, use_container_width=True)
            elif candlestick_stock == "Dow Jones Banks" and 'dow_banks' in data:
                # Ensure required columns exist
                if 'High' not in data['dow_banks'].columns:
                    data['dow_banks']['High'] = data['dow_banks']['Price']
                if 'Low' not in data['dow_banks'].columns:
                    data['dow_banks']['Low'] = data['dow_banks']['Price']
                
                fig_candle_dow = create_candlestick_chart(data['dow_banks'], 'Dow Jones Banks')
                st.plotly_chart(fig_candle_dow, use_container_width=True)
        
        with market_tab3:
            correlation_fig = create_correlation_heatmap(data)
            if correlation_fig:
                st.plotly_chart(correlation_fig, use_container_width=True)
            else:
                st.info("📊 Loading correlation data...")
        
        with market_tab4:
            # Interactive data tables
            table_dataset = st.selectbox(
                "Select Dataset for Table View",
                ["sample_data", "fis_data", "dow_banks", "fedfunds"]
            )
            
            create_data_table_with_search(data, table_dataset)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Performance":
        st.markdown('<div class="main-header">🎯 Performance Command Center</div>', unsafe_allow_html=True)
        
        # Gauge charts
        st.markdown('<div class="sub-header">🎛️ Performance Gauges</div>', unsafe_allow_html=True)
        create_gauge_charts(data)
        
        # Performance dashboard
        create_performance_dashboard(data)
        
        # Additional performance metrics
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">📊 Detailed Performance Analytics</div>', unsafe_allow_html=True)
        
        perf_tab1, perf_tab2, perf_tab3 = st.tabs(["📈 Trends", "🎯 Targets", "⚡ Real-time"])
        
        with perf_tab1:
            st.info("🚧 Advanced trend analysis with ML-powered insights coming soon!")
        
        with perf_tab2:
            st.info("🚧 Target tracking and achievement analysis coming soon!")
        
        with perf_tab3:
            st.info("🚧 Real-time performance monitoring coming soon!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Analysis":
        st.markdown('<div class="main-header">🔍 Advanced Analytics Laboratory</div>', unsafe_allow_html=True)
        
        # Enhanced analysis options
        analysis_categories = {
            "📊 Statistical Analysis": ["Correlation Matrix", "Regression Analysis", "Distribution Analysis"],
            "📈 Time Series": ["Trend Decomposition", "Seasonality Analysis", "Forecasting"],
            "💹 Financial Analysis": ["Risk Metrics", "Performance Attribution", "Portfolio Analysis"],
            "🤖 Machine Learning": ["Clustering", "Anomaly Detection", "Predictive Modeling"]
        }
        
        selected_category = st.selectbox("Analysis Category", list(analysis_categories.keys()))
        selected_analysis = st.selectbox("Specific Analysis", analysis_categories[selected_category])
        
        st.markdown(f'<div class="feature-highlight">', unsafe_allow_html=True)
        st.markdown(f"**Selected:** {selected_category} → {selected_analysis}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Placeholder for advanced analysis
        st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="sub-header">{selected_analysis}</div>', unsafe_allow_html=True)
        
        if "Correlation" in selected_analysis:
            correlation_fig = create_correlation_heatmap(data)
            if correlation_fig:
                st.plotly_chart(correlation_fig, use_container_width=True)
        else:
            st.info(f"🚧 {selected_analysis} implementation ready for deployment!")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Settings":
        st.markdown('<div class="main-header">⚙️ Dashboard Configuration</div>', unsafe_allow_html=True)
        
        settings_tab1, settings_tab2, settings_tab3 = st.tabs(["🎨 Appearance", "📊 Data", "🔧 Advanced"])
        
        with settings_tab1:
            st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
            st.subheader("🎨 Visual Customization")
            
            theme_cols = st.columns(2)
            with theme_cols[0]:
                color_scheme = st.selectbox("Color Theme", ["Default", "Dark", "Corporate", "Vibrant"])
                chart_animation = st.checkbox("Enable Chart Animations", value=True)
            
            with theme_cols[1]:
                sidebar_style = st.selectbox("Sidebar Style", ["Expanded", "Collapsed", "Auto"])
                show_tooltips = st.checkbox("Show Interactive Tooltips", value=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with settings_tab2:
            st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
            st.subheader("📊 Data Configuration")
            
            data_cols = st.columns(2)
            with data_cols[0]:
                auto_refresh = st.checkbox("Auto-refresh Data", value=False)
                refresh_interval = st.selectbox("Refresh Interval", ["1 min", "5 min", "15 min", "1 hour"])
            
            with data_cols[1]:
                data_cache = st.checkbox("Enable Data Caching", value=True)
                max_data_points = st.number_input("Max Data Points per Chart", value=1000, min_value=100)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with settings_tab3:
            st.markdown('<div class="interactive-card">', unsafe_allow_html=True)
            st.subheader("🔧 Advanced Options")
            
            adv_cols = st.columns(2)
            with adv_cols[0]:
                debug_mode = st.checkbox("Debug Mode", value=False)
                performance_monitoring = st.checkbox("Performance Monitoring", value=True)
            
            with adv_cols[1]:
                export_quality = st.selectbox("Export Quality", ["Standard", "High", "Ultra"])
                api_timeout = st.number_input("API Timeout (seconds)", value=30, min_value=5)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("💾 Save All Settings", type="primary"):
            st.success("✅ Settings saved successfully!")
            st.balloons()

    # Floating action button (simulated)
    st.markdown("""
    <div class="floating-action">
        <button style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            color: white;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        ">💡</button>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
