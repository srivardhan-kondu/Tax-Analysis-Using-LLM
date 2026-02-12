"""
Data upload and management UI for Tax Evasion Detection System
"""
import streamlit as st
import pandas as pd
from typing import Optional
import io

from data.data_processor import DataProcessor, load_csv
from data.data_validator import DataValidator


def show_csv_upload() -> Optional[pd.DataFrame]:
    """
    Show CSV upload interface
    FR3: Allow users to upload company financial data in CSV format
    """
    st.subheader("📁 Upload Company Financial Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with company financial data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} companies found.")
            
            # Show preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10))
                st.caption(f"Showing first 10 of {len(df)} rows")
            
            # Validate data
            validator = DataValidator()
            is_valid, errors, warnings = validator.validate_dataframe(df)
            
            # Show validation results
            if errors:
                st.error(f"❌ Validation Errors ({len(errors)}):")
                for error in errors[:5]:
                    st.write(f"- {error}")
                if len(errors) > 5:
                    st.write(f"... and {len(errors) - 5} more errors")
            
            if warnings:
                st.warning(f"⚠️ Validation Warnings ({len(warnings)}):")
                for warning in warnings[:5]:
                    st.write(f"- {warning}")
                if len(warnings) > 5:
                    st.write(f"... and {len(warnings) - 5} more warnings")
            
            if not errors and not warnings:
                st.success("✅ Data validation passed!")
            
            return df if not errors else None
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return None
    
    return None


def show_manual_entry() -> Optional[dict]:
    """
    Show manual data entry form
    FR5: Allow users to enter new company financial data manually for real-time analysis
    """
    st.subheader("✍️ Manual Data Entry")
    
    with st.form("manual_entry_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input(
                "Company Name *",
                help="Enter the company name"
            )
            
            sales = st.number_input(
                "Sales ($) *",
                min_value=0.0,
                value=1000000.0,
                step=10000.0,
                format="%.2f",
                help="Total sales in dollars"
            )
            
            profit_margin = st.number_input(
                "Profit Margin *",
                min_value=-1.0,
                max_value=1.0,
                value=0.20,
                step=0.01,
                format="%.4f",
                help="Profit margin as a decimal (e.g., 0.20 for 20%)"
            )
            
            debt_ratio = st.number_input(
                "Debt Ratio *",
                min_value=0.0,
                max_value=10.0,
                value=0.50,
                step=0.01,
                format="%.4f",
                help="Debt-to-asset ratio"
            )
        
        with col2:
            revenue_growth = st.number_input(
                "Revenue Growth *",
                min_value=-1.0,
                max_value=5.0,
                value=0.15,
                step=0.01,
                format="%.4f",
                help="Revenue growth rate as a decimal (e.g., 0.15 for 15%)"
            )
            
            employee_growth = st.number_input(
                "Employee Growth *",
                min_value=-1.0,
                max_value=5.0,
                value=0.12,
                step=0.01,
                format="%.4f",
                help="Employee growth rate as a decimal"
            )
            
            operating_expenses = st.number_input(
                "Operating Expenses ($) *",
                min_value=0.0,
                value=700000.0,
                step=10000.0,
                format="%.2f",
                help="Total operating expenses in dollars"
            )
            
            tax_to_revenue_ratio = st.number_input(
                "Tax-to-Revenue Ratio *",
                min_value=0.0,
                max_value=1.0,
                value=0.18,
                step=0.01,
                format="%.4f",
                help="Tax payments divided by total revenue"
            )
        
        submitted = st.form_submit_button("🚀 Analyze Company", type="primary")
        
        if submitted:
            if not company_name:
                st.error("❌ Company name is required")
                return None
            
            # Create data dictionary
            company_data = {
                'company_name': company_name,
                'sales': sales,
                'revenue_growth': revenue_growth,
                'profit_margin': profit_margin,
                'employee_growth': employee_growth,
                'debt_ratio': debt_ratio,
                'operating_expenses': operating_expenses,
                'tax_to_revenue_ratio': tax_to_revenue_ratio
            }
            
            # Validate
            validator = DataValidator()
            is_valid, errors, warnings = validator.validate_single_entry(company_data)
            
            if errors:
                st.error("❌ Validation Errors:")
                for error in errors:
                    st.write(f"- {error}")
                return None
            
            if warnings:
                st.warning("⚠️ Warnings:")
                for warning in warnings:
                    st.write(f"- {warning}")
            
            # Store in session state for immediate analysis
            st.session_state.manual_entry_data = company_data
            st.session_state.trigger_manual_analysis = True
            st.rerun()
    
    # Check if we need to run analysis (after rerun)
    if st.session_state.get('trigger_manual_analysis') and st.session_state.get('manual_entry_data'):
        st.session_state.trigger_manual_analysis = False
        return st.session_state.manual_entry_data
    
    return None


def show_sample_data_option():
    """Show option to use sample data"""
    st.subheader("🎲 Use Sample Data")
    
    st.info("Load pre-configured sample data with 20 companies for testing")
    
    if st.button("Load Sample Data", key="load_sample"):
        try:
            import config
            sample_file = config.DATA_DIR / "sample_companies.csv"
            
            if sample_file.exists():
                df = pd.read_csv(sample_file)
                st.success(f"✅ Loaded {len(df)} sample companies!")
                
                with st.expander("📊 Sample Data Preview"):
                    st.dataframe(df)
                
                return df
            else:
                st.error("❌ Sample data file not found")
                return None
        except Exception as e:
            st.error(f"❌ Error loading sample data: {str(e)}")
            return None
    
    return None


def export_results(results_df: pd.DataFrame, format: str = "csv"):
    """
    Export results
    FR23: Support export of analysis reports
    """
    if format == "csv":
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="tax_evasion_analysis_results.csv",
            mime="text/csv"
        )
    elif format == "excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Analysis Results')
        
        st.download_button(
            label="📥 Download Results as Excel",
            data=buffer.getvalue(),
            file_name="tax_evasion_analysis_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
# Data Upload Interface Module
