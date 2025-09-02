import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f1f3f4;
    }
    .stSelectbox > div > div > select {
        background-color: white;
        border: 2px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data files"""
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
        data['dow_banks']['Date'] = pd.to_datetime(data['dow_banks']['Date'], format='%d-%m-%Y')
        data['dow_banks']['Price'] = pd.to_numeric(data['dow_banks']['Price'], errors='coerce')
        data['dow_banks']['Open'] = pd.to_numeric(data['dow_banks']['Open'], errors='coerce')
    except Exception as e:
        st.error(f"Error loading Dow Jones Banks data: {e}")
    
    try:
        data['sp500'] = pd.read_csv('INDEX_US_S&P US_SPX.csv')
        # Handle the date format like "Jan-19" by adding "01-" prefix and converting to full year
        data['sp500']['Date'] = data['sp500']['Date'].apply(lambda x: f"01-{x}")
        data['sp500']['Date'] = pd.to_datetime(data['sp500']['Date'], format='%d-%b-%y')
        # Clean numeric columns by removing commas
        data['sp500']['Close'] = pd.to_numeric(data['sp500']['Close'].astype(str).str.replace(',', ''), errors='coerce')
        data['sp500']['Open'] = pd.to_numeric(data['sp500']['Open'].astype(str).str.replace(',', ''), errors='coerce')
    except Exception as e:
        st.error(f"Error loading S&P 500 data: {e}")
    
    # Load Excel files
    try:
        # Read the Excel file and check its actual structure
        unemployment_raw = pd.read_excel('Unemployment Rate.xlsx')
        # The file seems to have header information, let's find the actual data
        # Look for a row that contains "UNRATE" which is likely the data header
        data_start_row = None
        for i, row in unemployment_raw.iterrows():
            if any('UNRATE' in str(cell) for cell in row.values if pd.notna(cell)):
                data_start_row = i
                break
        
        if data_start_row is not None:
            # Read from the data start row
            data['unemployment'] = pd.read_excel('Unemployment Rate.xlsx', skiprows=data_start_row+1)
            # Clean up the data - usually FRED data has Date and Value columns
            data['unemployment'] = data['unemployment'].dropna()
            if len(data['unemployment'].columns) >= 2:
                data['unemployment'].columns = ['Date', 'Unemployment_Rate'] + list(data['unemployment'].columns[2:])
                data['unemployment']['Date'] = pd.to_datetime(data['unemployment']['Date'], errors='coerce')
                data['unemployment'] = data['unemployment'].dropna()
        else:
            # Fallback: try to read with different skip rows
            data['unemployment'] = pd.read_excel('Unemployment Rate.xlsx', skiprows=10)
            data['unemployment'] = data['unemployment'].dropna()
            if len(data['unemployment'].columns) >= 2:
                data['unemployment'].columns = ['Date', 'Unemployment_Rate'] + list(data['unemployment'].columns[2:])
                data['unemployment']['Date'] = pd.to_datetime(data['unemployment']['Date'], errors='coerce')
                data['unemployment'] = data['unemployment'].dropna()
    except Exception as e:
        st.error(f"Error loading Unemployment Rate data: {e}")
        # Create empty dataframe as fallback
        data['unemployment'] = pd.DataFrame(columns=['Date', 'Unemployment_Rate'])
    
    try:
        data['fedfunds'] = pd.read_excel('FEDFUNDS.xlsx')
        data['fedfunds']['Date'] = pd.to_datetime(data['fedfunds']['Date'])
    except Exception as e:
        st.error(f"Error loading Fed Funds data: {e}")
    
    try:
        # Similar approach for CPI data
        cpi_raw = pd.read_excel('Consumer Price Index.xlsx')
        # Look for actual data start
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
        # Create empty dataframe as fallback
        data['cpi'] = pd.DataFrame(columns=['Date', 'CPI'])
    
    return data

def create_sample_data_plot(df, level, selected_values):
    """Create hierarchical plot for sample data"""
    if level == 'Profile':
        grouped = df.groupby(['DATE', 'Profile'])['Actual'].sum().reset_index()
        fig = px.line(grouped, x='DATE', y='Actual', color='Profile',
                     title='Sample Data - Profile Level Analysis',
                     labels={'Actual': 'Actual Values', 'DATE': 'Date'})
    elif level == 'Line_Item':
        if selected_values['Profile']:
            filtered_df = df[df['Profile'].isin(selected_values['Profile'])]
            grouped = filtered_df.groupby(['DATE', 'Line_Item'])['Actual'].sum().reset_index()
            fig = px.line(grouped, x='DATE', y='Actual', color='Line_Item',
                         title=f'Sample Data - Line Item Level Analysis (Profile: {", ".join(selected_values["Profile"])})',
                         labels={'Actual': 'Actual Values', 'DATE': 'Date'})
        else:
            grouped = df.groupby(['DATE', 'Line_Item'])['Actual'].sum().reset_index()
            fig = px.line(grouped, x='DATE', y='Actual', color='Line_Item',
                         title='Sample Data - Line Item Level Analysis',
                         labels={'Actual': 'Actual Values', 'DATE': 'Date'})
    elif level == 'Site':
        filtered_df = df.copy()
        if selected_values['Profile']:
            filtered_df = filtered_df[filtered_df['Profile'].isin(selected_values['Profile'])]
        if selected_values['Line_Item']:
            filtered_df = filtered_df[filtered_df['Line_Item'].isin(selected_values['Line_Item'])]
        
        grouped = filtered_df.groupby(['DATE', 'Site'])['Actual'].sum().reset_index()
        fig = px.line(grouped, x='DATE', y='Actual', color='Site',
                     title='Sample Data - Site Level Analysis',
                     labels={'Actual': 'Actual Values', 'DATE': 'Date'})
    else:  # Lineup
        filtered_df = df.copy()
        if selected_values['Profile']:
            filtered_df = filtered_df[filtered_df['Profile'].isin(selected_values['Profile'])]
        if selected_values['Line_Item']:
            filtered_df = filtered_df[filtered_df['Line_Item'].isin(selected_values['Line_Item'])]
        if selected_values['Site']:
            filtered_df = filtered_df[filtered_df['Site'].isin(selected_values['Site'])]
        
        fig = px.line(filtered_df, x='DATE', y='Actual', color='Lineup',
                     title='Sample Data - Lineup Level Analysis (Most Detailed)',
                     labels={'Actual': 'Actual Values', 'DATE': 'Date'})
    
    fig.update_layout(height=500, showlegend=True)
    return fig

def create_comparison_plot(sample_data, plan_data):
    """Create comparison plot between sample data and plan data"""
    # Aggregate sample data
    sample_agg = sample_data.groupby('DATE')['Actual'].sum().reset_index()
    sample_agg['Type'] = 'Actual'
    sample_agg.rename(columns={'Actual': 'Value'}, inplace=True)
    
    # Aggregate plan data
    plan_agg = plan_data.groupby('DATE')['Plan'].sum().reset_index()
    plan_agg['Type'] = 'Plan'
    plan_agg.rename(columns={'Plan': 'Value'}, inplace=True)
    
    # Combine data
    combined = pd.concat([sample_agg, plan_agg])
    
    fig = px.line(combined, x='DATE', y='Value', color='Type',
                 title='Actual vs Plan Comparison',
                 labels={'Value': 'Amount', 'DATE': 'Date'})
    fig.update_layout(height=500)
    return fig

def create_stock_plot(data, title, price_col='Price'):
    """Create stock price plot"""
    fig = px.line(data, x='Date', y=price_col, title=title)
    fig.update_layout(height=400)
    return fig

def create_economic_indicators_plot(unemployment_data, fedfunds_data, cpi_data):
    """Create economic indicators plot"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Unemployment Rate (%)', 'Federal Funds Rate (%)', 'Consumer Price Index'),
        vertical_spacing=0.08
    )
    
    if unemployment_data is not None and not unemployment_data.empty and 'Date' in unemployment_data.columns:
        fig.add_trace(
            go.Scatter(x=unemployment_data['Date'], y=unemployment_data['Unemployment_Rate'],
                      mode='lines', name='Unemployment Rate', line=dict(color='red')),
            row=1, col=1
        )
    
    if fedfunds_data is not None and not fedfunds_data.empty and 'Date' in fedfunds_data.columns:
        fig.add_trace(
            go.Scatter(x=fedfunds_data['Date'], y=fedfunds_data['Federal Fund Rate'],
                      mode='lines', name='Fed Funds Rate', line=dict(color='blue')),
            row=2, col=1
        )
    
    if cpi_data is not None and not cpi_data.empty and 'Date' in cpi_data.columns:
        fig.add_trace(
            go.Scatter(x=cpi_data['Date'], y=cpi_data['CPI'],
                      mode='lines', name='CPI', line=dict(color='green')),
            row=3, col=1
        )
    
    fig.update_layout(height=800, title_text="Economic Indicators Overview")
    return fig

def main():
    # Load data
    data = load_data()
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox("Select Page", ["Data Overview", "Analysis"])
    
    if page == "Data Overview":
        st.markdown('<div class="main-header">📈 Data Overview Dashboard</div>', unsafe_allow_html=True)
        
        # Sample Data Section with Hierarchical Analysis
        st.markdown('<div class="sub-header">🎯 Sample Data Hierarchical Analysis</div>', unsafe_allow_html=True)
        
        if 'sample_data' in data:
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Hierarchy Controls")
                level = st.selectbox("Select Analysis Level", 
                                   ['Profile', 'Line_Item', 'Site', 'Lineup'],
                                   help="Profile → Line_Item → Site → Lineup (most detailed)")
                
                # Dynamic filters based on level
                selected_values = {'Profile': [], 'Line_Item': [], 'Site': [], 'Lineup': []}
                
                if level in ['Line_Item', 'Site', 'Lineup']:
                    profiles = data['sample_data']['Profile'].unique()
                    selected_values['Profile'] = st.multiselect("Filter by Profile", profiles)
                
                if level in ['Site', 'Lineup'] and selected_values['Profile']:
                    filtered_df = data['sample_data'][data['sample_data']['Profile'].isin(selected_values['Profile'])]
                    line_items = filtered_df['Line_Item'].unique()
                    selected_values['Line_Item'] = st.multiselect("Filter by Line Item", line_items)
                
                if level == 'Lineup':
                    if selected_values['Line_Item']:
                        filtered_df = data['sample_data'][
                            (data['sample_data']['Profile'].isin(selected_values['Profile'])) &
                            (data['sample_data']['Line_Item'].isin(selected_values['Line_Item']))
                        ]
                        sites = filtered_df['Site'].unique()
                        selected_values['Site'] = st.multiselect("Filter by Site", sites)
            
            with col2:
                fig = create_sample_data_plot(data['sample_data'], level, selected_values)
                st.plotly_chart(fig, use_container_width=True)
        
        # Sample Data vs Plan Comparison
        st.markdown('<div class="sub-header">📊 Actual vs Plan Comparison</div>', unsafe_allow_html=True)
        if 'sample_data' in data and 'plan_data' in data:
            fig_comparison = create_comparison_plot(data['sample_data'], data['plan_data'])
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Stock Data Section
        st.markdown('<div class="sub-header">💰 Financial Markets Overview</div>', unsafe_allow_html=True)
        
        stock_cols = st.columns(2)
        
        with stock_cols[0]:
            if 'fis_data' in data:
                fig_fis = create_stock_plot(data['fis_data'], 'FIS Stock Price', 'Price')
                st.plotly_chart(fig_fis, use_container_width=True)
            
            if 'nasdaq_tech' in data:
                fig_nasdaq_tech = create_stock_plot(data['nasdaq_tech'], 'NASDAQ 100 Technology Sector', 'Price')
                st.plotly_chart(fig_nasdaq_tech, use_container_width=True)
        
        with stock_cols[1]:
            if 'kbw_nasdaq' in data:
                fig_kbw = create_stock_plot(data['kbw_nasdaq'], 'KBW NASDAQ Financial Technology', 'Price')
                st.plotly_chart(fig_kbw, use_container_width=True)
            
            if 'dow_banks' in data:
                fig_dow = create_stock_plot(data['dow_banks'], 'Dow Jones Banks', 'Price')
                st.plotly_chart(fig_dow, use_container_width=True)
        
        # S&P 500 and Economic Indicators
        if 'sp500' in data:
            fig_sp500 = create_stock_plot(data['sp500'], 'S&P 500 Index', 'Close')
            st.plotly_chart(fig_sp500, use_container_width=True)
        
        # Economic Indicators
        st.markdown('<div class="sub-header">🏛️ Economic Indicators</div>', unsafe_allow_html=True)
        if any(key in data for key in ['unemployment', 'fedfunds', 'cpi']):
            fig_econ = create_economic_indicators_plot(
                data.get('unemployment'), 
                data.get('fedfunds'), 
                data.get('cpi')
            )
            st.plotly_chart(fig_econ, use_container_width=True)
    
    elif page == "Analysis":
        st.markdown('<div class="main-header">🔍 Advanced Analysis</div>', unsafe_allow_html=True)
        
        st.info("📋 Advanced analysis features will be implemented here. This page is ready for your specific analysis requirements.")
        
        # Placeholder for analysis features
        analysis_options = st.selectbox(
            "Select Analysis Type",
            ["Correlation Analysis", "Trend Analysis", "Performance Metrics", "Risk Assessment", "Forecasting"]
        )
        
        if analysis_options == "Correlation Analysis":
            st.subheader("📈 Correlation Analysis")
            st.write("Correlation analysis between different datasets will be implemented here.")
        
        elif analysis_options == "Trend Analysis":
            st.subheader("📊 Trend Analysis")
            st.write("Trend analysis and pattern recognition will be implemented here.")
        
        elif analysis_options == "Performance Metrics":
            st.subheader("🎯 Performance Metrics")
            st.write("Performance metrics and KPI analysis will be implemented here.")
        
        elif analysis_options == "Risk Assessment":
            st.subheader("⚠️ Risk Assessment")
            st.write("Risk assessment and volatility analysis will be implemented here.")
        
        elif analysis_options == "Forecasting":
            st.subheader("🔮 Forecasting")
            st.write("Forecasting models and predictions will be implemented here.")

if __name__ == "__main__":
    main()
