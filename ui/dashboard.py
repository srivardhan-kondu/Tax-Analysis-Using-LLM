"""
Dashboard UI for Tax Evasion Detection System
Displays results and visualizations
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List


def is_valid_value(value):
    """
    Safely check if a value is not null/empty.
    Handles scalars, lists, arrays, and pandas types.
    """
    if value is None:
        return False
    if isinstance(value, (list, np.ndarray)):
        return len(value) > 0
    if isinstance(value, str):
        return len(value.strip()) > 0
    try:
        return not pd.isna(value)
    except (ValueError, TypeError):
        # If pd.isna fails (e.g., for arrays), assume value is valid
        return True


def show_risk_summary(summary: Dict):
    """
    Display risk summary statistics
    FR20: Display results in clear and interpretable format
    """
    st.subheader("📊 Analysis Summary")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Companies",
            summary['total_companies']
        )
    
    with col2:
        high_risk = summary['risk_distribution']['high_risk']
        st.metric(
            "High Risk",
            high_risk['count'],
            delta=f"{high_risk['percentage']}%",
            delta_color="inverse"
        )
    
    with col3:
        medium_risk = summary['risk_distribution']['medium_risk']
        st.metric(
            "Medium Risk",
            medium_risk['count'],
            delta=f"{medium_risk['percentage']}%"
        )
    
    with col4:
        st.metric(
            "Avg Confidence",
            f"{summary['average_confidence']:.1%}"
        )
    
    # Risk distribution chart
    st.subheader("🎯 Risk Distribution")
    
    risk_data = pd.DataFrame([
        {"Risk Level": "No Risk", "Count": summary['risk_distribution']['no_risk']['count'], "Color": "#28a745"},
        {"Risk Level": "Medium Risk", "Count": summary['risk_distribution']['medium_risk']['count'], "Color": "#ffc107"},
        {"Risk Level": "High Risk", "Count": summary['risk_distribution']['high_risk']['count'], "Color": "#dc3545"},
    ])
    
    if risk_data['Count'].sum() > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = px.pie(
                risk_data,
                values='Count',
                names='Risk Level',
                color='Risk Level',
                color_discrete_map={
                    'No Risk': '#28a745',
                    'Medium Risk': '#ffc107',
                    'High Risk': '#dc3545'
                },
                title="Risk Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Bar chart
            fig = px.bar(
                risk_data,
                x='Risk Level',
                y='Count',
                color='Risk Level',
                color_discrete_map={
                    'No Risk': '#28a745',
                    'Medium Risk': '#ffc107',
                    'High Risk': '#dc3545'
                },
                title="Companies by Risk Level"
            )
            st.plotly_chart(fig, use_container_width=True)


def show_results_table(results_df: pd.DataFrame, show_details: bool = True):
    """
    Display results in a table format
    FR20: Display tax risk results in clear format
    """
    st.subheader("📋 Detailed Results")
    
    # Prepare display dataframe
    display_columns = [
        'company_name',
        'risk_level',
        'risk_category',
        'confidence_score'
    ]
    
    if all(col in results_df.columns for col in display_columns):
        display_df = results_df[display_columns].copy()
        
        # Format confidence score
        display_df['confidence_score'] = display_df['confidence_score'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
        
        # Rename columns for display
        display_df.columns = ['Company', 'Risk Level', 'Risk Category', 'Confidence']
        
        # Color code risk levels
        def highlight_risk(row):
            if row['Risk Level'] == 2:
                return ['background-color: #ffcccc'] * len(row)
            elif row['Risk Level'] == 1:
                return ['background-color: #fff3cd'] * len(row)
            elif row['Risk Level'] == 0:
                return ['background-color: #d4edda'] * len(row)
            return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_risk, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Show detailed view for selected company
        if show_details and len(display_df) > 0:
            show_company_details(results_df)


def show_company_details(results_df: pd.DataFrame):
    """
    Show detailed analysis for a selected company
    FR21: Highlight key financial factors
    """
    st.subheader("🔍 Company Details")
    
    # Get unique company names to avoid duplicates in dropdown
    company_names = results_df['company_name'].unique().tolist()
    selected_company = st.selectbox(
        "Select a company to view details:",
        company_names
    )
    
    if selected_company:
        # Filter and ensure we get only the first matching row
        filtered_df = results_df[results_df['company_name'] == selected_company]
        company_data = filtered_df.iloc[0].to_dict()  # Convert to dict to avoid Series issues
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Analysis", "📈 Financial Metrics"])
        
        with tab1:
            show_company_overview(company_data)
        
        with tab2:
            show_company_analysis(company_data)
        
        with tab3:
            show_financial_metrics(company_data)


def show_company_overview(company_data: dict):
    """Show company overview"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Assessment")
        risk_level = company_data.get('risk_level', -1)
        risk_category = company_data.get('risk_category', 'Unknown')
        confidence = company_data.get('confidence_score', 0)
        
        # Risk level badge
        if risk_level == 0:
            st.success(f"✅ {risk_category}")
        elif risk_level == 1:
            st.warning(f"⚠️ {risk_category}")
        elif risk_level == 2:
            st.error(f"🚨 {risk_category}")
        else:
            st.info(f"❓ {risk_category}")
        
        st.metric("Confidence Score", f"{confidence:.1%}")
        
        if company_data.get('needs_manual_review', False):
            st.warning("⚠️ Manual review recommended")
    
    with col2:
        st.markdown("### Key Information")
        st.write(f"**Company:** {company_data.get('company_name', 'N/A')}")
        
        timestamp = company_data.get('analysis_timestamp')
        if is_valid_value(timestamp):
            st.write(f"**Analyzed:** {timestamp}")


def show_company_analysis(company_data: dict):
    """Show detailed analysis"""
    # Key findings
    findings = company_data.get('key_findings')
    if is_valid_value(findings):
        st.markdown("### 🎯 Key Findings")
        if isinstance(findings, list):
            for finding in findings:
                st.write(f"- {finding}")
        else:
            st.write(str(findings))
    
    # Red flags
    red_flags = company_data.get('red_flags')
    if is_valid_value(red_flags) and isinstance(red_flags, list) and len(red_flags) > 0:
        st.markdown("### 🚩 Red Flags")
        for flag in red_flags:
            st.error(f"⚠️ {flag}")
    
    # Reasoning
    reasoning = company_data.get('reasoning')
    if is_valid_value(reasoning):
        st.markdown("### 💡 Detailed Reasoning")
        st.info(str(reasoning))


def show_financial_metrics(company_data: dict):
    """Show financial metrics visualization"""
    st.markdown("### Financial Indicators")
    
    # Create metrics dictionary
    metrics = {}
    metric_columns = [
        'sales', 'revenue_growth', 'profit_margin', 'employee_growth',
        'debt_ratio', 'operating_expenses', 'tax_to_revenue_ratio'
    ]
    
    for col in metric_columns:
        value = company_data.get(col)
        if is_valid_value(value):
            try:
                metrics[col.replace('_', ' ').title()] = float(value)
            except (ValueError, TypeError):
                pass
    
    if metrics:
        # Display as columns
        cols = st.columns(3)
        for idx, (metric_name, value) in enumerate(metrics.items()):
            with cols[idx % 3]:
                # Format percentage values
                if any(term in metric_name.lower() for term in ['growth', 'margin', 'ratio']):
                    st.metric(metric_name, f"{value:.2%}")
                else:
                    st.metric(metric_name, f"${value:,.2f}")


def show_high_risk_companies(results_df: pd.DataFrame):
    """
    Show prioritized list of high-risk companies
    Helps with audit prioritization
    """
    st.subheader("🚨 High-Risk Companies (Audit Priority)")
    
    high_risk = results_df[results_df['risk_level'] == 2].copy()
    
    if len(high_risk) > 0:
        # Sort by confidence score
        high_risk = high_risk.sort_values('confidence_score', ascending=False)
        
        st.write(f"**{len(high_risk)} companies identified as high-risk**")
        
        # Display in expandable cards
        for idx, row in high_risk.iterrows():
            row_dict = row.to_dict()  # Convert to dict to avoid Series issues
            with st.expander(
                f"🏢 {row_dict.get('company_name', 'Unknown')} - Confidence: {row_dict.get('confidence_score', 0):.1%}"
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    sales = row_dict.get('sales')
                    if is_valid_value(sales):
                        st.write(f"**Sales:** ${float(sales):,.2f}")
                    profit_margin = row_dict.get('profit_margin')
                    if is_valid_value(profit_margin):
                        st.write(f"**Profit Margin:** {float(profit_margin):.2%}")
                    tax_ratio = row_dict.get('tax_to_revenue_ratio')
                    if is_valid_value(tax_ratio):
                        st.write(f"**Tax-to-Revenue Ratio:** {float(tax_ratio):.2%}")
                
                with col2:
                    findings = row_dict.get('key_findings')
                    if is_valid_value(findings) and isinstance(findings, list):
                        st.write("**Key Issues:**")
                        for finding in findings[:3]:
                            st.write(f"- {finding}")
    else:
        st.success("✅ No high-risk companies found!")


def show_confidence_distribution(results_df: pd.DataFrame):
    """Show confidence score distribution"""
    st.subheader("📊 Confidence Score Distribution")
    
    if 'confidence_score' in results_df.columns:
        fig = px.histogram(
            results_df,
            x='confidence_score',
            nbins=20,
            title="Distribution of Confidence Scores",
            labels={'confidence_score': 'Confidence Score'},
            color_discrete_sequence=['#17a2b8']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
# Dashboard UI Module
