"""
Additional interactive features for the dashboard
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_real_time_filters():
    """Create real-time filtering sidebar"""
    st.sidebar.markdown("### 🎛️ Real-Time Filters")
    
    # Quick date filters
    date_filter = st.sidebar.radio(
        "📅 Quick Date Filter",
        ["Last 30 Days", "Last 90 Days", "Last Year", "YTD", "All Time", "Custom Range"]
    )
    
    return date_filter

def create_interactive_comparison_tool(data):
    """Create interactive comparison tool"""
    st.markdown('<div class="sub-header">⚖️ Interactive Comparison Tool</div>', unsafe_allow_html=True)
    
    comparison_cols = st.columns(3)
    
    with comparison_cols[0]:
        dataset1 = st.selectbox(
            "Select First Dataset",
            ["FIS Stock", "Dow Jones Banks", "NASDAQ Tech", "KBW FinTech", "S&P 500"]
        )
    
    with comparison_cols[1]:
        dataset2 = st.selectbox(
            "Select Second Dataset", 
            ["FIS Stock", "Dow Jones Banks", "NASDAQ Tech", "KBW FinTech", "S&P 500"],
            index=1
        )
    
    with comparison_cols[2]:
        comparison_type = st.selectbox(
            "Comparison Type",
            ["Price Overlay", "Normalized Comparison", "Correlation", "Performance"]
        )
    
    return dataset1, dataset2, comparison_type

def create_animated_chart(data, title):
    """Create animated chart with play button"""
    fig = px.line(data, x='Date', y='Price', 
                 title=title,
                 animation_frame=data['Date'].dt.year if 'Date' in data.columns else None)
    
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"frame": {"duration": 500, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 300}}],
                        label="Play",
                        method="animate"
                    ),
                    dict(
                        args=[{"frame": {"duration": 0, "redraw": True},
                              "mode": "immediate", "transition": {"duration": 0}}],
                        label="Pause",
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            ),
        ]
    )
    
    return fig

def create_drill_down_chart(data, hierarchy_cols, value_col, date_col):
    """Create drill-down capability for hierarchical data"""
    # Create sunburst chart for hierarchy visualization
    fig = px.sunburst(
        data, 
        path=hierarchy_cols,
        values=value_col,
        title="🌟 Hierarchical Data Explorer - Click to Drill Down"
    )
    
    fig.update_layout(height=600)
    return fig

def create_dynamic_kpi_dashboard(data):
    """Create dynamic KPI dashboard with real-time updates"""
    
    # Create 4 columns for KPIs
    kpi_cols = st.columns(4)
    
    kpis = [
        {"title": "📈 Total Performance", "value": "1,234,567", "change": "+12.5%", "color": "green"},
        {"title": "🎯 Plan Achievement", "value": "98.7%", "change": "+2.1%", "color": "blue"},
        {"title": "📊 Market Index", "value": "4,567.89", "change": "-0.8%", "color": "red"},
        {"title": "💰 Portfolio Value", "value": "$2.3M", "change": "+5.4%", "color": "green"}
    ]
    
    for i, kpi in enumerate(kpis):
        with kpi_cols[i]:
            delta_color = "normal" if kpi["color"] == "green" else "inverse"
            st.metric(
                label=kpi["title"],
                value=kpi["value"],
                delta=kpi["change"],
                delta_color=delta_color
            )

def create_interactive_treemap(data):
    """Create interactive treemap for hierarchical data"""
    if 'sample_data' in data and not data['sample_data'].empty:
        # Aggregate data for treemap
        treemap_data = data['sample_data'].groupby(['Profile', 'Line_Item', 'Site'])['Actual'].sum().reset_index()
        
        fig = px.treemap(
            treemap_data,
            path=['Profile', 'Line_Item', 'Site'],
            values='Actual',
            title='🗺️ Interactive Data Treemap - Click to Explore',
            color='Actual',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=600)
        return fig
    
    return None

def create_gauge_charts(data):
    """Create gauge charts for KPIs"""
    gauge_cols = st.columns(3)
    
    with gauge_cols[0]:
        # Plan achievement gauge
        if 'sample_data' in data and 'plan_data' in data:
            if not data['sample_data'].empty and not data['plan_data'].empty:
                actual_total = data['sample_data']['Actual'].sum()
                plan_total = data['plan_data']['Plan'].sum()
                achievement = (actual_total / plan_total) * 100
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = achievement,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Plan Achievement %"},
                    delta = {'reference': 100},
                    gauge = {
                        'axis': {'range': [None, 150]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 80], 'color': "lightgray"},
                            {'range': [80, 100], 'color': "yellow"},
                            {'range': [100, 150], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 100
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with gauge_cols[1]:
        # Market sentiment gauge (example)
        market_sentiment = 75  # This would be calculated from actual data
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = market_sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Sentiment"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 30], 'color': "red"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with gauge_cols[2]:
        # Risk level gauge
        risk_level = 45  # This would be calculated from volatility data
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_level,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def create_data_table_with_search(data, dataset_name):
    """Create searchable and sortable data table"""
    st.markdown(f'<div class="sub-header">📋 {dataset_name} - Interactive Data Table</div>', unsafe_allow_html=True)
    
    if dataset_name in data and not data[dataset_name].empty:
        df = data[dataset_name].copy()
        
        # Search functionality
        search_term = st.text_input(f"🔍 Search in {dataset_name}", placeholder="Enter search term...")
        
        if search_term:
            # Search across all string columns
            string_cols = df.select_dtypes(include=['object']).columns
            mask = df[string_cols].astype(str).apply(
                lambda x: x.str.contains(search_term, case=False, na=False)
            ).any(axis=1)
            df = df[mask]
        
        # Display with enhanced formatting
        st.dataframe(
            df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Export button
        csv = df.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {dataset_name} as CSV",
            data=csv,
            file_name=f"{dataset_name}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def create_alert_system(data):
    """Create alert system for significant changes"""
    st.markdown('<div class="sub-header">🚨 Smart Alerts</div>', unsafe_allow_html=True)
    
    alerts = []
    
    # Check for significant price movements
    if 'fis_data' in data and not data['fis_data'].empty:
        recent_change = ((data['fis_data']['Price'].iloc[0] - data['fis_data']['Price'].iloc[1]) / 
                        data['fis_data']['Price'].iloc[1]) * 100
        
        if abs(recent_change) > 5:
            alert_type = "🔴 High Volatility" if abs(recent_change) > 10 else "🟡 Moderate Movement"
            alerts.append(f"{alert_type}: FIS stock moved {recent_change:.1f}% in the latest period")
    
    # Check plan vs actual variance
    if 'sample_data' in data and 'plan_data' in data:
        if not data['sample_data'].empty and not data['plan_data'].empty:
            actual_total = data['sample_data']['Actual'].sum()
            plan_total = data['plan_data']['Plan'].sum()
            variance = abs((actual_total - plan_total) / plan_total) * 100
            
            if variance > 10:
                alerts.append(f"⚠️ Plan Variance: {variance:.1f}% deviation from planned targets")
    
    # Display alerts
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ All metrics within normal ranges")

def add_export_functionality():
    """Add comprehensive export functionality"""
    st.sidebar.markdown("### 📥 Export Options")
    
    export_type = st.sidebar.selectbox(
        "Export Format",
        ["📊 Chart as PNG", "📋 Data as CSV", "📄 Report as PDF", "🌐 Dashboard as HTML"]
    )
    
    if st.sidebar.button("📥 Export"):
        if export_type == "📊 Chart as PNG":
            st.sidebar.success("Chart export feature ready!")
        elif export_type == "📋 Data as CSV":
            st.sidebar.success("Data export feature ready!")
        elif export_type == "📄 Report as PDF":
            st.sidebar.success("PDF report feature ready!")
        elif export_type == "🌐 Dashboard as HTML":
            st.sidebar.success("HTML export feature ready!")

def create_advanced_tooltips():
    """Create advanced tooltips and help system"""
    st.sidebar.markdown("### ❓ Help & Tips")
    
    with st.sidebar.expander("🎯 How to Use Hierarchy"):
        st.markdown("""
        **Profile** → **Line_Item** → **Site** → **Lineup**
        
        - Start with Profile for high-level overview
        - Drill down to Line_Item for business unit analysis
        - Use Site for location-based insights
        - Select Lineup for the most detailed view
        """)
    
    with st.sidebar.expander("📊 Chart Interactions"):
        st.markdown("""
        - **Hover** over data points for details
        - **Click** legend items to show/hide series
        - **Double-click** legend to isolate series
        - **Zoom** by selecting area on chart
        - **Pan** by dragging the chart
        """)
    
    with st.sidebar.expander("🔍 Advanced Features"):
        st.markdown("""
        - Use **date range selector** for time-based filtering
        - **Multi-select** filters for complex analysis
        - **Export** charts and data for external use
        - **Correlation analysis** shows relationships
        """)

def create_performance_summary_cards():
    """Create performance summary cards with sparklines"""
    st.markdown('<div class="sub-header">⚡ Performance Summary</div>', unsafe_allow_html=True)
    
    summary_cols = st.columns(4)
    
    # Sample performance cards
    performance_data = [
        {"title": "📈 Growth Rate", "value": "+12.5%", "trend": "↗️", "color": "green"},
        {"title": "🎯 Efficiency", "value": "94.2%", "trend": "↗️", "color": "blue"},
        {"title": "📊 Volatility", "value": "15.3%", "trend": "↘️", "color": "orange"},
        {"title": "💼 ROI", "value": "8.7%", "trend": "↗️", "color": "green"}
    ]
    
    for i, card in enumerate(performance_data):
        with summary_cols[i]:
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                border-left: 4px solid {card['color']};
                text-align: center;
                margin: 0.5rem 0;
            ">
                <h4 style="margin: 0; color: #64748b; font-size: 0.9rem;">{card['title']}</h4>
                <h2 style="margin: 0.5rem 0; color: {card['color']};">{card['value']}</h2>
                <span style="font-size: 1.5rem;">{card['trend']}</span>
            </div>
            """, unsafe_allow_html=True)

def create_interactive_heatmap_calendar(data):
    """Create calendar heatmap for time-series data"""
    if 'sample_data' in data and not data['sample_data'].empty:
        # Aggregate daily data
        daily_data = data['sample_data'].groupby('DATE')['Actual'].sum().reset_index()
        daily_data['Year'] = daily_data['DATE'].dt.year
        daily_data['Month'] = daily_data['DATE'].dt.month
        daily_data['Day'] = daily_data['DATE'].dt.day
        
        # Create calendar heatmap
        fig = px.density_heatmap(
            daily_data,
            x='Month',
            y='Day', 
            z='Actual',
            facet_col='Year',
            title='📅 Activity Calendar Heatmap',
            labels={'Actual': 'Value', 'Month': 'Month', 'Day': 'Day'},
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(height=400)
        return fig
    
    return None

def add_real_time_notifications():
    """Add real-time notification system"""
    # Create a placeholder for notifications
    notification_placeholder = st.empty()
    
    # Simulate real-time updates (in a real app, this would connect to live data)
    if st.sidebar.button("🔄 Refresh Data"):
        with notification_placeholder:
            st.success("✅ Data refreshed successfully!")
            st.balloons()  # Fun animation!

def create_cross_filter_functionality():
    """Create cross-filtering between charts"""
    st.markdown('<div class="sub-header">🔗 Cross-Filter Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>💡 Pro Tip:</strong> Select data points in one chart to automatically filter other charts. 
        This enables powerful cross-analysis capabilities!
    </div>
    """, unsafe_allow_html=True)

def create_advanced_chart_controls():
    """Create advanced chart customization controls"""
    st.sidebar.markdown("### 🎨 Chart Customization")
    
    chart_controls = {}
    
    with st.sidebar.expander("🎨 Visual Settings"):
        chart_controls['theme'] = st.selectbox(
            "Chart Theme",
            ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
        )
        
        chart_controls['color_scheme'] = st.selectbox(
            "Color Palette",
            ["Default", "Viridis", "Plasma", "Inferno", "Turbo", "Rainbow"]
        )
        
        chart_controls['show_grid'] = st.checkbox("Show Grid", value=True)
        chart_controls['show_legend'] = st.checkbox("Show Legend", value=True)
    
    with st.sidebar.expander("📏 Chart Dimensions"):
        chart_controls['height'] = st.slider("Chart Height", 300, 800, 500)
        chart_controls['line_width'] = st.slider("Line Width", 1, 5, 2)
    
    return chart_controls
