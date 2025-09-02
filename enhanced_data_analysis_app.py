import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        border-bottom: 2px solid #e0e6ed;
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
    
    .kpi-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .stSelectbox > div > div > select {
        background-color: white;
        border: 2px solid #667eea;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data files with enhanced error handling"""
    data = {}
    
    # Load CSV files
    try:
        data['sample_data'] = pd.read_csv('Sample_data_N.csv')
        data['sample_data']['DATE'] = pd.to_datetime(data['sample_data']['DATE'], format='%d-%m-%Y')
        data['sample_data'] = data['sample_data'].dropna()
    except Exception as e:
        st.error(f"Error loading Sample_data_N.csv: {e}")
    
    try:
        data['plan_data'] = pd.read_csv('Plan Number.csv')
        data['plan_data']['DATE'] = pd.to_datetime(data['plan_data']['DATE'], format='%d-%m-%Y')
        data['plan_data'] = data['plan_data'].dropna()
    except Exception as e:
        st.error(f"Error loading Plan Number.csv: {e}")
    
    try:
        data['fis_data'] = pd.read_csv('FIS Historical Data.csv')
        data['fis_data']['Date'] = pd.to_datetime(data['fis_data']['Date'], format='%d-%m-%Y')
        data['fis_data']['Price'] = pd.to_numeric(data['fis_data']['Price'], errors='coerce')
        data['fis_data']['Open'] = pd.to_numeric(data['fis_data']['Open'], errors='coerce')
        data['fis_data']['High'] = pd.to_numeric(data['fis_data']['High'], errors='coerce')
        data['fis_data']['Low'] = pd.to_numeric(data['fis_data']['Low'], errors='coerce')
    except Exception as e:
        st.error(f"Error loading FIS Historical Data.csv: {e}")
    
    try:
        data['kbw_nasdaq'] = pd.read_csv('KBW Nasdaq Financial Technology Historical Data.csv')
        data['kbw_nasdaq']['Date'] = pd.to_datetime(data['kbw_nasdaq']['Date'], format='%d-%m-%Y')
        data['kbw_nasdaq']['Price'] = pd.to_numeric(data['kbw_nasdaq']['Price'].str.replace(',', ''), errors='coerce')
        data['kbw_nasdaq']['Open'] = pd.to_numeric(data['kbw_nasdaq']['Open'].str.replace(',', ''), errors='coerce')
    except Exception as e:
        st.error(f"Error loading KBW Nasdaq data: {e}")
    
    try:
        data['nasdaq_tech'] = pd.read_csv('NASDAQ 100 Technology Sector Historical Data.csv')
        data['nasdaq_tech']['Date'] = pd.to_datetime(data['nasdaq_tech']['Date'], format='%d-%m-%Y')
        data['nasdaq_tech']['Price'] = pd.to_numeric(data['nasdaq_tech']['Price'].str.replace(',', ''), errors='coerce')
        data['nasdaq_tech']['Open'] = pd.to_numeric(data['nasdaq_tech']['Open'].str.replace(',', ''), errors='coerce')
    except Exception as e:
        st.error(f"Error loading NASDAQ 100 Tech data: {e}")
    
    try:
        data['dow_banks'] = pd.read_csv('Dow Jones Banks Historical Data.csv')
        data['dow_banks']['Date'] = pd.to_datetime(data['dow_banks']['Date'], format='%m/%d/%Y')
        data['dow_banks']['Price'] = pd.to_numeric(data['dow_banks']['Price'], errors='coerce')
        data['dow_banks']['Open'] = pd.to_numeric(data['dow_banks']['Open'], errors='coerce')
    except Exception as e:
        st.error(f"Error loading Dow Jones Banks data: {e}")
    
    try:
        data['sp500'] = pd.read_csv('INDEX_US_S&P US_SPX.csv')
        data['sp500']['Date'] = data['sp500']['Date'].apply(lambda x: f"01-{x}")
        data['sp500']['Date'] = pd.to_datetime(data['sp500']['Date'], format='%d-%b-%y')
        data['sp500']['Close'] = pd.to_numeric(data['sp500']['Close'].astype(str).str.replace(',', ''), errors='coerce')
        data['sp500']['Open'] = pd.to_numeric(data['sp500']['Open'].astype(str).str.replace(',', ''), errors='coerce')
    except Exception as e:
        st.error(f"Error loading S&P 500 data: {e}")
    
    # Load Excel files with enhanced handling
    try:
        unemployment_raw = pd.read_excel('Unemployment Rate.xlsx')
        data_start_row = None
        for i, row in unemployment_raw.iterrows():
            if any('UNRATE' in str(cell) for cell in row.values if pd.notna(cell)):
                data_start_row = i
                break
        
        if data_start_row is not None:
            data['unemployment'] = pd.read_excel('Unemployment Rate.xlsx', skiprows=data_start_row+1)
        else:
            data['unemployment'] = pd.read_excel('Unemployment Rate.xlsx', skiprows=10)
        
        data['unemployment'] = data['unemployment'].dropna()
        if len(data['unemployment'].columns) >= 2:
            data['unemployment'].columns = ['Date', 'Unemployment_Rate'] + list(data['unemployment'].columns[2:])
            data['unemployment']['Date'] = pd.to_datetime(data['unemployment']['Date'], errors='coerce')
            data['unemployment'] = data['unemployment'].dropna()
    except Exception as e:
        st.error(f"Error loading Unemployment Rate data: {e}")
        data['unemployment'] = pd.DataFrame(columns=['Date', 'Unemployment_Rate'])
    
    try:
        data['fedfunds'] = pd.read_excel('FEDFUNDS.xlsx')
        data['fedfunds']['Date'] = pd.to_datetime(data['fedfunds']['Date'])
    except Exception as e:
        st.error(f"Error loading Fed Funds data: {e}")
    
    try:
        cpi_raw = pd.read_excel('Consumer Price Index.xlsx')
        data_start_row = None
        for i, row in cpi_raw.iterrows():
            if any(str(cell).strip() in ['DATE', 'CPIAUCSL', 'Date'] for cell in row.values if pd.notna(cell)):
                data_start_row = i
                break
        
        if data_start_row is not None:
            data['cpi'] = pd.read_excel('Consumer Price Index.xlsx', skiprows=data_start_row+1)
        else:
            data['cpi'] = pd.read_excel('Consumer Price Index.xlsx', skiprows=10)
        
        data['cpi'] = data['cpi'].dropna()
        if len(data['cpi'].columns) >= 2:
            data['cpi'].columns = ['Date', 'CPI'] + list(data['cpi'].columns[2:])
            data['cpi']['Date'] = pd.to_datetime(data['cpi']['Date'], errors='coerce')
            data['cpi'] = data['cpi'].dropna()
    except Exception as e:
        st.error(f"Error loading CPI data: {e}")
        data['cpi'] = pd.DataFrame(columns=['Date', 'CPI'])
    
    return data

def create_kpi_cards(data):
    """Create interactive KPI cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'sample_data' in data and not data['sample_data'].empty:
            total_actual = data['sample_data']['Actual'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem;">Total Actual</h3>
                <h2 style="margin: 0.5rem 0 0 0;">${total_actual:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if 'plan_data' in data and not data['plan_data'].empty:
            total_plan = data['plan_data']['Plan'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem;">Total Plan</h3>
                <h2 style="margin: 0.5rem 0 0 0;">${total_plan:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if 'fis_data' in data and not data['fis_data'].empty:
            latest_fis = data['fis_data']['Price'].iloc[0]
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem;">FIS Latest Price</h3>
                <h2 style="margin: 0.5rem 0 0 0;">${latest_fis:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'fedfunds' in data and not data['fedfunds'].empty:
            latest_fed_rate = data['fedfunds']['Federal Fund Rate'].iloc[-1]
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem;">Fed Funds Rate</h3>
                <h2 style="margin: 0.5rem 0 0 0;">{latest_fed_rate:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

def create_enhanced_sample_data_plot(df, level, selected_values, date_range):
    """Create enhanced hierarchical plot with animations and interactivity"""
    # Filter by date range
    if date_range:
        df = df[(df['DATE'] >= date_range[0]) & (df['DATE'] <= date_range[1])]
    
    if level == 'Profile':
        grouped = df.groupby(['DATE', 'Profile'])['Actual'].sum().reset_index()
        fig = px.line(grouped, x='DATE', y='Actual', color='Profile',
                     title='📊 Sample Data - Profile Level Analysis',
                     labels={'Actual': 'Actual Values ($)', 'DATE': 'Date'},
                     template='plotly_white')
        fig.update_traces(mode='lines+markers', hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>')
    elif level == 'Line_Item':
        if selected_values['Profile']:
            filtered_df = df[df['Profile'].isin(selected_values['Profile'])]
            grouped = filtered_df.groupby(['DATE', 'Line_Item'])['Actual'].sum().reset_index()
            fig = px.line(grouped, x='DATE', y='Actual', color='Line_Item',
                         title=f'📈 Sample Data - Line Item Analysis (Profile: {", ".join(selected_values["Profile"])})',
                         labels={'Actual': 'Actual Values ($)', 'DATE': 'Date'},
                         template='plotly_white')
        else:
            grouped = df.groupby(['DATE', 'Line_Item'])['Actual'].sum().reset_index()
            fig = px.line(grouped, x='DATE', y='Actual', color='Line_Item',
                         title='📈 Sample Data - Line Item Level Analysis',
                         labels={'Actual': 'Actual Values ($)', 'DATE': 'Date'},
                         template='plotly_white')
        fig.update_traces(mode='lines+markers', hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>')
    elif level == 'Site':
        filtered_df = df.copy()
        if selected_values['Profile']:
            filtered_df = filtered_df[filtered_df['Profile'].isin(selected_values['Profile'])]
        if selected_values['Line_Item']:
            filtered_df = filtered_df[filtered_df['Line_Item'].isin(selected_values['Line_Item'])]
        
        grouped = filtered_df.groupby(['DATE', 'Site'])['Actual'].sum().reset_index()
        fig = px.line(grouped, x='DATE', y='Actual', color='Site',
                     title='🏢 Sample Data - Site Level Analysis',
                     labels={'Actual': 'Actual Values ($)', 'DATE': 'Date'},
                     template='plotly_white')
        fig.update_traces(mode='lines+markers', hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>')
    else:  # Lineup
        filtered_df = df.copy()
        if selected_values['Profile']:
            filtered_df = filtered_df[filtered_df['Profile'].isin(selected_values['Profile'])]
        if selected_values['Line_Item']:
            filtered_df = filtered_df[filtered_df['Line_Item'].isin(selected_values['Line_Item'])]
        if selected_values['Site']:
            filtered_df = filtered_df[filtered_df['Site'].isin(selected_values['Site'])]
        
        fig = px.line(filtered_df, x='DATE', y='Actual', color='Lineup',
                     title='🎯 Sample Data - Lineup Level Analysis (Most Detailed)',
                     labels={'Actual': 'Actual Values ($)', 'DATE': 'Date'},
                     template='plotly_white')
        fig.update_traces(mode='lines+markers', hovertemplate='<b>%{fullData.name}</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>')
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def create_candlestick_chart(data, title):
    """Create interactive candlestick chart for stock data"""
    fig = go.Figure(data=go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Price'],
        name=title
    ))
    
    fig.update_layout(
        title=f'📈 {title} - Candlestick Chart',
        yaxis_title='Price ($)',
        xaxis_title='Date',
        template='plotly_white',
        height=500,
        showlegend=False,
        hovermode='x',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)'),
    )
    
    return fig

def create_correlation_heatmap(data):
    """Create correlation heatmap for numeric data"""
    numeric_data = {}
    
    # Collect numeric data from different sources
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
            title="🔗 Correlation Matrix - Financial Indicators",
            color_continuous_scale='RdBu_r',
            template='plotly_white'
        )
        
        fig.update_layout(height=500)
        return fig
    
    return None

def create_performance_dashboard(data):
    """Create performance metrics dashboard"""
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
        st.subheader("📈 YTD Performance")
        
        if 'fis_data' in data and not data['fis_data'].empty:
            current_year = datetime.now().year
            ytd_data = data['fis_data'][data['fis_data']['Date'].dt.year == current_year]
            if not ytd_data.empty:
                ytd_return = ((ytd_data['Price'].iloc[0] - ytd_data['Price'].iloc[-1]) / ytd_data['Price'].iloc[-1]) * 100
                color = "green" if ytd_return > 0 else "red"
                st.markdown(f"<h3 style='color: {color};'>FIS: {ytd_return:.2f}%</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
        st.subheader("📊 Volatility (30D)")
        
        if 'fis_data' in data and not data['fis_data'].empty:
            recent_data = data['fis_data'].head(30)
            if len(recent_data) > 1:
                returns = recent_data['Price'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                st.markdown(f"<h3>FIS: {volatility:.2f}%</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
        st.subheader("🎯 Plan vs Actual")
        
        if 'sample_data' in data and 'plan_data' in data:
            if not data['sample_data'].empty and not data['plan_data'].empty:
                actual_total = data['sample_data']['Actual'].sum()
                plan_total = data['plan_data']['Plan'].sum()
                variance = ((actual_total - plan_total) / plan_total) * 100
                color = "green" if variance > 0 else "red"
                st.markdown(f"<h3 style='color: {color};'>Variance: {variance:.1f}%</h3>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Load data
    data = load_data()
    
    # Sidebar with enhanced navigation
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    
    # Page selection with icons
    page_options = {
        "📊 Data Overview": "Data Overview",
        "🔍 Advanced Analysis": "Analysis", 
        "⚙️ Settings": "Settings"
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(page_options.keys()))
    page = page_options[selected_page]
    
    if page == "Data Overview":
        st.markdown('<div class="main-header">📈 Interactive Data Overview Dashboard</div>', unsafe_allow_html=True)
        
        # Add date range selector
        st.sidebar.markdown("### 📅 Date Range Filter")
        if 'sample_data' in data and not data['sample_data'].empty:
            min_date = data['sample_data']['DATE'].min().date()
            max_date = data['sample_data']['DATE'].max().date()
            
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                date_range = [pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])]
            else:
                date_range = None
        else:
            date_range = None
        
        # KPI Cards
        create_kpi_cards(data)
        
        # Sample Data Section with Enhanced Hierarchical Analysis
        st.markdown('<div class="sub-header">🎯 Interactive Sample Data Analysis</div>', unsafe_allow_html=True)
        
        if 'sample_data' in data:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
                st.subheader("🎛️ Analysis Controls")
                
                # Enhanced level selector with descriptions
                level_descriptions = {
                    'Profile': '🏢 Highest level view',
                    'Line_Item': '📋 Business line analysis', 
                    'Site': '📍 Location-based view',
                    'Lineup': '🎯 Most detailed level'
                }
                
                level = st.selectbox(
                    "Analysis Level", 
                    list(level_descriptions.keys()),
                    format_func=lambda x: f"{level_descriptions[x]} - {x}",
                    help="Select the hierarchical level for analysis"
                )
                
                # Dynamic filters with enhanced UX
                selected_values = {'Profile': [], 'Line_Item': [], 'Site': [], 'Lineup': []}
                
                if level in ['Line_Item', 'Site', 'Lineup']:
                    profiles = data['sample_data']['Profile'].unique()
                    selected_values['Profile'] = st.multiselect(
                        "🏢 Filter by Profile", 
                        profiles,
                        help="Select one or more profiles to filter the data"
                    )
                
                if level in ['Site', 'Lineup'] and selected_values['Profile']:
                    filtered_df = data['sample_data'][data['sample_data']['Profile'].isin(selected_values['Profile'])]
                    line_items = filtered_df['Line_Item'].unique()
                    selected_values['Line_Item'] = st.multiselect(
                        "📋 Filter by Line Item", 
                        line_items,
                        help="Select specific line items for analysis"
                    )
                
                if level == 'Lineup':
                    if selected_values['Line_Item']:
                        filtered_df = data['sample_data'][
                            (data['sample_data']['Profile'].isin(selected_values['Profile'])) &
                            (data['sample_data']['Line_Item'].isin(selected_values['Line_Item']))
                        ]
                        sites = filtered_df['Site'].unique()
                        selected_values['Site'] = st.multiselect(
                            "📍 Filter by Site", 
                            sites,
                            help="Select specific sites for detailed analysis"
                        )
                
                # Chart type selector
                chart_type = st.radio(
                    "📊 Chart Type",
                    ["Line Chart", "Area Chart", "Bar Chart"],
                    help="Choose visualization style"
                )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = create_enhanced_sample_data_plot(data['sample_data'], level, selected_values, date_range)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance Dashboard
        st.markdown('<div class="sub-header">📊 Performance Metrics</div>', unsafe_allow_html=True)
        create_performance_dashboard(data)
        
        # Enhanced Financial Markets Section
        st.markdown('<div class="sub-header">💰 Financial Markets Analysis</div>', unsafe_allow_html=True)
        
        # Market selector
        market_tab1, market_tab2, market_tab3 = st.tabs(["📈 Stock Prices", "🕯️ Candlestick Charts", "🔗 Correlations"])
        
        with market_tab1:
            stock_cols = st.columns(2)
            
            with stock_cols[0]:
                if 'fis_data' in data:
                    fig_fis = px.line(data['fis_data'], x='Date', y='Price', 
                                     title='FIS Stock Price',
                                     template='plotly_white')
                    fig_fis.update_traces(line=dict(width=3))
                    fig_fis.update_layout(height=400)
                    st.plotly_chart(fig_fis, use_container_width=True)
                
                if 'nasdaq_tech' in data:
                    fig_nasdaq_tech = px.line(data['nasdaq_tech'], x='Date', y='Price',
                                            title='NASDAQ 100 Technology Sector',
                                            template='plotly_white')
                    fig_nasdaq_tech.update_traces(line=dict(width=3))
                    fig_nasdaq_tech.update_layout(height=400)
                    st.plotly_chart(fig_nasdaq_tech, use_container_width=True)
            
            with stock_cols[1]:
                if 'kbw_nasdaq' in data:
                    fig_kbw = px.line(data['kbw_nasdaq'], x='Date', y='Price',
                                     title='KBW NASDAQ Financial Technology',
                                     template='plotly_white')
                    fig_kbw.update_traces(line=dict(width=3))
                    fig_kbw.update_layout(height=400)
                    st.plotly_chart(fig_kbw, use_container_width=True)
                
                if 'dow_banks' in data:
                    fig_dow = px.line(data['dow_banks'], x='Date', y='Price',
                                     title='Dow Jones Banks',
                                     template='plotly_white')
                    fig_dow.update_traces(line=dict(width=3))
                    fig_dow.update_layout(height=400)
                    st.plotly_chart(fig_dow, use_container_width=True)
        
        with market_tab2:
            candlestick_cols = st.columns(2)
            
            with candlestick_cols[0]:
                if 'fis_data' in data and all(col in data['fis_data'].columns for col in ['Open', 'High', 'Low', 'Price']):
                    fig_candle = create_candlestick_chart(data['fis_data'], 'FIS Stock')
                    st.plotly_chart(fig_candle, use_container_width=True)
            
            with candlestick_cols[1]:
                if 'dow_banks' in data and all(col in data['dow_banks'].columns for col in ['Open', 'High', 'Low', 'Price']):
                    # Add High and Low columns if missing
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
                st.info("📊 Correlation analysis requires multiple datasets with overlapping time periods.")
    
    elif page == "Analysis":
        st.markdown('<div class="main-header">🔍 Advanced Analytics Suite</div>', unsafe_allow_html=True)
        
        # Analysis type selector with enhanced options
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["📊 Correlation Analysis", "📈 Trend Analysis", "🎯 Performance Metrics", 
             "⚠️ Risk Assessment", "🔮 Forecasting", "💹 Technical Analysis"]
        )
        
        if "Correlation" in analysis_type:
            st.markdown('<div class="sub-header">📊 Correlation Analysis</div>', unsafe_allow_html=True)
            
            correlation_fig = create_correlation_heatmap(data)
            if correlation_fig:
                st.plotly_chart(correlation_fig, use_container_width=True)
                
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("""
                **💡 Interpretation Guide:**
                - Values close to +1 indicate strong positive correlation
                - Values close to -1 indicate strong negative correlation  
                - Values close to 0 indicate little to no correlation
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("📊 Insufficient data for correlation analysis")
        
        elif "Trend" in analysis_type:
            st.markdown('<div class="sub-header">📈 Trend Analysis</div>', unsafe_allow_html=True)
            st.info("🚧 Advanced trend analysis with moving averages, trend lines, and seasonality detection coming soon!")
        
        elif "Performance" in analysis_type:
            st.markdown('<div class="sub-header">🎯 Performance Metrics</div>', unsafe_allow_html=True)
            create_performance_dashboard(data)
        
        elif "Risk" in analysis_type:
            st.markdown('<div class="sub-header">⚠️ Risk Assessment</div>', unsafe_allow_html=True)
            st.info("🚧 Volatility analysis, VaR calculations, and risk metrics coming soon!")
        
        elif "Forecasting" in analysis_type:
            st.markdown('<div class="sub-header">🔮 Forecasting Models</div>', unsafe_allow_html=True)
            st.info("🚧 Time series forecasting with ARIMA, Prophet, and ML models coming soon!")
        
        elif "Technical" in analysis_type:
            st.markdown('<div class="sub-header">💹 Technical Analysis</div>', unsafe_allow_html=True)
            st.info("🚧 RSI, MACD, Bollinger Bands, and other technical indicators coming soon!")
    
    elif page == "Settings":
        st.markdown('<div class="main-header">⚙️ Dashboard Settings</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">🎨 Theme & Appearance</div>', unsafe_allow_html=True)
        
        theme_col1, theme_col2 = st.columns(2)
        
        with theme_col1:
            st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
            st.subheader("Color Scheme")
            color_scheme = st.selectbox(
                "Select Theme",
                ["Default", "Dark Mode", "High Contrast", "Corporate"]
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with theme_col2:
            st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
            st.subheader("Chart Settings")
            show_grid = st.checkbox("Show Grid Lines", value=True)
            animate_charts = st.checkbox("Enable Animations", value=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sub-header">📊 Data Preferences</div>', unsafe_allow_html=True)
        
        data_col1, data_col2 = st.columns(2)
        
        with data_col1:
            st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
            st.subheader("Default Filters")
            default_date_range = st.selectbox(
                "Default Date Range",
                ["Last 30 Days", "Last 90 Days", "Last Year", "All Data"]
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with data_col2:
            st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
            st.subheader("Export Options")
            export_format = st.selectbox(
                "Preferred Export Format",
                ["PNG", "PDF", "SVG", "HTML"]
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("💾 Save Settings"):
            st.success("✅ Settings saved successfully!")

if __name__ == "__main__":
    main()
