"""
Main Streamlit Application for Tax Evasion Detection System
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import config
from ui.authentication import (
    show_login_page,
    is_logged_in,
    require_login,
    show_user_info,
    is_admin,
    save_user_analysis,
    load_user_analysis,
    load_all_users_analysis,
    get_all_usernames
)
from ui.data_upload import (
    show_csv_upload,
    show_manual_entry,
    show_sample_data_option,
    export_results
)
from ui.dashboard import (
    show_risk_summary,
    show_results_table,
    show_high_risk_companies,
    show_confidence_distribution
)
from ui.ml_dashboard import show_ml_dashboard
from engine.risk_predictor import RiskPredictor
from data.data_processor import DataProcessor
from ml.adaboost_model import TaxEvasionAdaBoost
from ml.training_data import generate_training_data, get_feature_columns


# Page config
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = None
    if 'ml_trained' not in st.session_state:
        st.session_state.ml_trained = False


def get_ml_model():
    """Get or auto-train ML model"""
    if st.session_state.get('ml_model') is not None and st.session_state.get('ml_trained'):
        return st.session_state.ml_model
    
    import os
    model_path = os.path.join(project_root, 'data', 'adaboost_model.joblib')
    model = TaxEvasionAdaBoost(n_estimators=100, learning_rate=0.8)
    
    if os.path.exists(model_path):
        if model.load_model():
            st.session_state.ml_model = model
            st.session_state.ml_trained = True
            return model
    
    # Auto-train if no model exists
    df = generate_training_data(1000)
    X = df[get_feature_columns()]
    y = df['risk_label'].values
    model.train(X, y)
    model.save_model()
    st.session_state.ml_model = model
    st.session_state.ml_trained = True
    return model


def show_sidebar():
    """Show sidebar navigation"""
    with st.sidebar:
        st.title("🔍 Tax Evasion Detection")
        st.markdown("---")
        
        if is_logged_in():
            show_user_info()
            st.markdown("---")
            
            # Navigation
            page = st.radio(
                "Navigation",
                ["🏠 Home", "📊 Analyze Companies", "🤖 LLM ", "📈 View Results"],
                label_visibility="collapsed"
            )
            
            return page
        else:
            return "login"


def show_home_page():
    """Show home page"""
    st.title("🏠 Welcome to Tax Evasion Detection System")
    
    st.markdown("""
    This intelligent system uses **GPT-4.0 AI** to analyze company financial data and identify 
    potential tax evasion patterns.
    
    ### 🎯 Key Features
    
    - **📁 CSV Upload**: Upload multiple companies' financial data at once
    - **✍️ Manual Entry**: Analyze individual companies in real-time
    - **🤖 AI-Powered Analysis**: Uses GPT-4.0 with anti-hallucination measures
    - **📊 Risk Classification**: Categorizes companies into No Risk, Medium Risk, or High Risk
    - **📈 Detailed Reports**: Get comprehensive analysis with key findings
    - **💾 Export Results**: Download analysis reports in CSV/Excel format
    
    ### 🔐 Security
    
    - Secure authentication required
    - All predictions are validated with rule-based checks
    - Confidence scoring for every prediction
    
    ### 🚀 Get Started
    
    Navigate to **"📊 Analyze Companies"** to begin your analysis!
    """)
    
    # Show statistics if available
    if st.session_state.analysis_results is not None:
        st.markdown("---")
        st.subheader("📊 Latest Analysis")
        results_df = st.session_state.analysis_results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Companies Analyzed", len(results_df))
        with col2:
            high_risk = len(results_df[results_df['risk_level'] == 2])
            st.metric("High Risk", high_risk)
        with col3:
            avg_conf = results_df['confidence_score'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1%}")


def show_analyze_page():
    """Show analysis page"""
    st.title("📊 Analyze Company Financial Data")
    
    # Check for API key
    if not config.OPENAI_API_KEY:
        st.error("""
        ❌ **OpenAI API Key not configured!**
        
        Please set your OpenAI API key in the `.env` file:
        1. Copy `.env.example` to `.env`
        2. Add your OpenAI API key: `OPENAI_API_KEY=your-key-here`
        3. Restart the application
        """)
        st.stop()
    
    # Analysis mode selection
    analysis_mode = st.radio(
        "Choose Analysis Mode:",
        ["📁 Upload CSV File", "✍️ Manual Entry", "🎲 Use Sample Data"],
        horizontal=True
    )
    
    st.markdown("---")
    
    data_to_analyze = None
    is_batch = False
    run_analysis_immediately = False  # Flag for manual entry
    
    if analysis_mode == "📁 Upload CSV File":
        data_to_analyze = show_csv_upload()
        is_batch = True
    
    elif analysis_mode == "✍️ Manual Entry":
        data_to_analyze = show_manual_entry()
        is_batch = False
        # Manual entry should run immediately when data is returned
        run_analysis_immediately = data_to_analyze is not None
    
    elif analysis_mode == "🎲 Use Sample Data":
        data_to_analyze = show_sample_data_option()
        is_batch = True
    
    # Analyze button (for batch modes) or immediate analysis (for manual entry)
    if data_to_analyze is not None:
        st.markdown("---")
        
        # For manual entry, run immediately; for batch modes, require button click
        should_analyze = run_analysis_immediately or st.button("🚀 Start Analysis", type="primary", use_container_width=True)
        
        if should_analyze:
            with st.spinner("Analyzing... This may take a moment..."):
                try:
                    # Initialize predictor
                    predictor = RiskPredictor()
                    
                    if is_batch:
                        # Batch analysis with BOTH ML and LLM
                        results_df = predictor.predict_batch(data_to_analyze, preprocess=True)
                        
                        # Add ML predictions
                        ml_model = get_ml_model()
                        ml_predictions = []
                        for _, row in data_to_analyze.iterrows():
                            company_data = row.to_dict()
                            ml_result = ml_model.predict_single(company_data)
                            ml_predictions.append({
                                'ml_risk_level': ml_result['risk_level'],
                                'ml_risk_category': ml_result['risk_category'],
                                'ml_confidence': ml_result['confidence']
                            })
                        ml_df = pd.DataFrame(ml_predictions)
                        results_df = pd.concat([results_df.reset_index(drop=True), ml_df], axis=1)
                        
                        # Store in session state and save to user profile
                        st.session_state.analysis_results = results_df
                        st.session_state.current_data = data_to_analyze
                        
                        # Save to user's profile for persistence
                        username = st.session_state.get('username')
                        if username:
                            save_user_analysis(username, results_df)
                        
                        st.success(f"✅ Analysis complete! Analyzed {len(results_df)} companies with ML + LLM.")
                        
                        # Show summary
                        summary = predictor.get_risk_summary(results_df)
                        show_risk_summary(summary)
                        
                        # Show AdaBoost vs LLM comparison
                        st.markdown("### 🔄 AdaBoost ML vs Neural Networks Comparison")
                        comparison_df = results_df[['company_name', 'risk_category', 'ml_risk_category', 'confidence_score', 'ml_confidence']].copy()
                        comparison_df.columns = ['Company', 'LLM Prediction', 'AdaBoost Prediction', 'LLM Confidence', 'AdaBoost Confidence']
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # Show results table
                        show_results_table(results_df, show_details=False)
                        
                        # Export option
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            export_results(results_df, format="csv")
                        with col2:
                            export_results(results_df, format="excel")
                    
                    else:
                        # Single analysis (Manual Entry) with BOTH ML and LLM
                        assessment = predictor.predict_single(data_to_analyze, validate_input=True)
                        
                        # Get ML prediction
                        ml_model = get_ml_model()
                        ml_result = ml_model.predict_single(data_to_analyze)
                        
                        # Store in session state as single-row DataFrame for View Results page
                        import datetime
                        assessment['analysis_timestamp'] = datetime.datetime.now().isoformat()
                        assessment['ml_risk_level'] = ml_result['risk_level']
                        assessment['ml_risk_category'] = ml_result['risk_category']
                        assessment['ml_confidence'] = ml_result['confidence']
                        results_df = pd.DataFrame([assessment])
                        st.session_state.analysis_results = results_df
                        
                        # Save to user's profile for persistence
                        username = st.session_state.get('username')
                        if username:
                            save_user_analysis(username, results_df)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader(f"📋 Analysis Results for {assessment['company_name']}")
                        
                        # Show BOTH AdaBoost ML and LLM predictions
                        st.markdown("### 🔄 Dual Prediction (AdaBoost ML + Neural Networks)")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 🤖 AdaBoost ML")
                            ml_risk = ml_result['risk_level']
                            ml_cat = ml_result['risk_category']
                            if ml_risk == 0:
                                st.success(f"✅ **{ml_cat}**")
                            elif ml_risk == 1:
                                st.warning(f"⚠️ **{ml_cat}**")
                            else:
                                st.error(f"🚨 **{ml_cat}**")
                            st.metric("ML Confidence", f"{ml_result['confidence']:.1%}")
                        
                        with col2:
                            st.markdown("#### 🧠 Neural Networks")
                            risk_level = assessment['risk_level']
                            risk_category = assessment['risk_category']
                            if risk_level == 0:
                                st.success(f"✅ **{risk_category}**")
                            elif risk_level == 1:
                                st.warning(f"⚠️ **{risk_category}**")
                            elif risk_level == 2:
                                st.error(f"🚨 **{risk_category}**")
                            else:
                                st.info(f"❓ **{risk_category}**")
                            st.metric("LLM Confidence", f"{assessment['confidence_score']:.1%}")
                        
                        # Agreement indicator
                        if ml_result['risk_level'] == assessment['risk_level']:
                            st.success("✅ **Both ML and LLM agree on the risk level!**")
                        else:
                            st.warning("⚠️ **ML and LLM predictions differ - recommend further review.**")
                        
                        # Key findings (from LLM)
                        if 'key_findings' in assessment and assessment['key_findings']:
                            st.markdown("### 🎯 Key Findings (LLM Analysis)")
                            if isinstance(assessment['key_findings'], list):
                                for finding in assessment['key_findings']:
                                    st.write(f"- {finding}")
                            else:
                                st.write(assessment['key_findings'])
                        
                        # Red flags
                        if 'red_flags' in assessment and assessment['red_flags'] and len(assessment['red_flags']) > 0:
                            st.markdown("### 🚩 Red Flags")
                            for flag in assessment['red_flags']:
                                st.error(f"⚠️ {flag}")
                        
                        # Detailed reasoning
                        if 'reasoning' in assessment and assessment['reasoning']:
                            with st.expander("💡 Detailed LLM Reasoning"):
                                st.info(assessment['reasoning'])
                        
                        # Clear the manual entry trigger after successful analysis
                        if 'manual_entry_data' in st.session_state:
                            del st.session_state.manual_entry_data
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.exception(e)


def show_results_page():
    """Show results page with user-specific or admin access"""
    st.title("📈 Analysis Results")
    
    username = st.session_state.get('username')
    user_is_admin = is_admin()
    
    # Admin view - select user or all
    if user_is_admin:
        st.info("👑 **Admin Mode** - You can view any user's data")
        
        all_users = get_all_usernames()
        view_options = ["All Users"] + all_users
        
        selected_view = st.selectbox(
            "View data for:",
            view_options,
            key="admin_user_select"
        )
        
        if selected_view == "All Users":
            results_df = load_all_users_analysis()
        else:
            results_df = load_user_analysis(selected_view)
    else:
        # Regular user - only their own data
        results_df = load_user_analysis(username)
    
    # Also include current session results if available
    if st.session_state.get('analysis_results') is not None:
        session_results = st.session_state.analysis_results
        if results_df is not None:
            # Don't duplicate - session results are already saved
            pass
        else:
            results_df = session_results
    
    if results_df is None or len(results_df) == 0:
        st.info("ℹ️ No analysis results available. Please analyze some companies first!")
        if st.button("Go to Analysis Page"):
            st.rerun()
        return
    
    predictor = RiskPredictor()
    
    # Show summary
    summary = predictor.get_risk_summary(results_df)
    show_risk_summary(summary)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 All Results",
        "🚨 High Risk Companies",
        "📊 Confidence Analysis",
        "💾 Export"
    ])
    
    with tab1:
        # Show who analyzed (for admin)
        if user_is_admin and 'analyzed_by' in results_df.columns:
            st.caption("💡 'Analyzed By' column shows which user performed each analysis")
        show_results_table(results_df, show_details=True)
    
    with tab2:
        show_high_risk_companies(results_df)
    
    with tab3:
        show_confidence_distribution(results_df)
    
    with tab4:
        st.subheader("💾 Export Results")
        col1, col2 = st.columns(2)
        with col1:
            export_results(results_df, format="csv")
        with col2:
            export_results(results_df, format="excel")



def main():
    """Main application entry point"""
    initialize_session_state()
    
    # Check if user is logged in
    if not is_logged_in():
        show_login_page()
        return
    
    # Show sidebar and get selected page
    page = show_sidebar()
    
    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Analyze Companies":
        require_login()
        show_analyze_page()
    elif page == "🤖 LLM ":
        require_login()
        show_ml_dashboard()
    elif page == "📈 View Results":
        require_login()
        show_results_page()


if __name__ == "__main__":
    main()
# Author: Srivardhan Kondu
