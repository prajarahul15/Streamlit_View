# Financial Data Analysis Dashboard

A comprehensive Streamlit-based dashboard for analyzing financial and economic data with interactive visualizations and hierarchical data exploration.

## Features

### 📊 Data Overview Page
- **Hierarchical Sample Data Analysis**: Interactive dropdown system for exploring data at different levels (Profile → Line_Item → Site → Lineup)
- **Actual vs Plan Comparison**: Side-by-side comparison of actual performance against planned targets
- **Financial Markets Overview**: Multiple stock and index visualizations including:
  - FIS Stock Data
  - KBW NASDAQ Financial Technology Index
  - NASDAQ 100 Technology Sector
  - Dow Jones Banks Index
  - S&P 500 Index
- **Economic Indicators**: Unemployment Rate, Federal Funds Rate, and Consumer Price Index trends

### 🔍 Analysis Page
- Framework for advanced analysis features (ready for implementation):
  - Correlation Analysis
  - Trend Analysis
  - Performance Metrics
  - Risk Assessment
  - Forecasting

## Data Files Supported

### CSV Files
- `Sample_data_N.csv` - Base sample data with hierarchical structure
- `Plan Number.csv` - 2025 plan data for comparison
- `FIS Historical Data.csv` - FIS stock price data
- `KBW Nasdaq Financial Technology Historical Data.csv` - KBW NASDAQ FinTech index
- `NASDAQ 100 Technology Sector Historical Data.csv` - NASDAQ 100 tech sector data
- `Dow Jones Banks Historical Data.csv` - Dow Jones banking sector data
- `INDEX_US_S&P US_SPX.csv` - S&P 500 index data

### Excel Files
- `Unemployment Rate.xlsx` - US unemployment rate data
- `FEDFUNDS.xlsx` - Federal funds rate data
- `Consumer Price Index.xlsx` - CPI inflation data

## Installation & Setup

1. **Create Virtual Environment** (if not already created):
   ```bash
   python -m venv FIS
   ```

2. **Activate Virtual Environment**:
   ```bash
   # Windows
   FIS\Scripts\activate
   
   # Linux/Mac
   source FIS/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install streamlit plotly pandas openpyxl numpy
   ```

## Usage

1. **Run the Application**:
   ```bash
   streamlit run data_analysis_app.py
   ```

2. **Navigate the Dashboard**:
   - Use the sidebar to switch between "Data Overview" and "Analysis" pages
   - In Data Overview, explore the hierarchical sample data using the dropdown controls
   - Select different analysis levels (Profile, Line_Item, Site, Lineup) for detailed insights
   - Apply filters to focus on specific profiles, line items, or sites

## Data Hierarchy

The sample data follows this hierarchical structure:
```
Profile (Top Level)
├── Line_Item
    ├── Site
        └── Lineup (Most Detailed Level)
```

### Interactive Features
- **Profile Level**: Shows aggregated data across all profiles
- **Line_Item Level**: Drill down by line items, optionally filtered by profile
- **Site Level**: Analyze by site, with optional profile and line item filters
- **Lineup Level**: Most detailed view with full filtering capabilities

## Technical Details

### Data Processing
- Automatic date parsing and formatting
- Numeric data cleaning and conversion
- Missing data handling
- Data aggregation for hierarchical views

### Visualization
- Interactive Plotly charts
- Responsive design for different screen sizes
- Custom styling with modern UI elements
- Real-time filtering and updates

### Performance
- Data caching for faster load times
- Efficient data processing
- Optimized chart rendering

## Error Handling
- Graceful handling of missing or corrupted data files
- User-friendly error messages
- Fallback options for incomplete datasets

## Future Enhancements
- Advanced correlation analysis
- Predictive modeling and forecasting
- Export functionality for charts and data
- Real-time data connections
- Custom dashboard configurations

## Support
For issues or questions, please refer to the data file formats and ensure all required files are present in the application directory.
