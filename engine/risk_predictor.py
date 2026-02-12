"""
Risk Prediction Engine for Tax Evasion Detection
Orchestrates the analysis process using GPT-4 and validation
"""
import logging
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime

from ai.gpt4_analyzer import GPT4Analyzer
from ai.validation import OutputValidator
from data.data_processor import DataProcessor
from data.data_validator import DataValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskPredictor:
    """
    Main prediction engine that orchestrates tax evasion risk analysis
    Implements FR16-FR19: Risk prediction and classification
    """
    
    def __init__(self):
        """Initialize the risk predictor with all components"""
        self.gpt_analyzer = GPT4Analyzer()
        self.output_validator = OutputValidator()
        self.data_processor = DataProcessor()
        self.data_validator = DataValidator()
        
        logger.info("Risk Predictor initialized successfully")
    
    def predict_single(self, company_data: Dict, validate_input: bool = True) -> Dict:
        """
        Predict tax evasion risk for a single company
        
        Args:
            company_data: Dictionary containing company financial data
            validate_input: Whether to validate input data first
        
        Returns:
            Risk assessment dictionary
        """
        logger.info(f"Predicting risk for: {company_data.get('company_name', 'Unknown')}")
        
        # Step 1: Validate input (FR5)
        if validate_input:
            is_valid, errors, warnings = self.data_validator.validate_single_entry(company_data)
            if not is_valid:
                logger.error(f"Input validation failed: {errors}")
                return {
                    'company_name': company_data.get('company_name', 'Unknown'),
                    'risk_level': -1,
                    'risk_category': 'Invalid Data',
                    'confidence_score': 0.0,
                    'errors': errors,
                    'warnings': warnings,
                    'needs_manual_review': True
                }
        
        # Step 2: Analyze with GPT-4 (FR16, FR17)
        try:
            assessment = self.gpt_analyzer.analyze_company(company_data)
        except Exception as e:
            logger.error(f"GPT-4 analysis failed: {str(e)}")
            return {
                'company_name': company_data.get('company_name', 'Unknown'),
                'risk_level': -1,
                'risk_category': 'Analysis Failed',
                'confidence_score': 0.0,
                'error': str(e),
                'needs_manual_review': True
            }
        
        # Step 3: Validate output (anti-hallucination)
        is_valid, validation_issues = self.output_validator.validate_assessment(
            assessment, company_data
        )
        
        if validation_issues:
            assessment['validation_issues'] = validation_issues
            logger.warning(f"Validation issues found: {validation_issues}")
        
        # Step 4: Add metadata
        assessment['prediction_timestamp'] = datetime.now().isoformat()
        assessment['model_used'] = self.gpt_analyzer.model
        
        logger.info(
            f"Prediction complete: {assessment['company_name']} - "
            f"Risk Level {assessment['risk_level']} ({assessment['risk_category']})"
        )
        
        return assessment
    
    def predict_batch(self, companies_df: pd.DataFrame, preprocess: bool = True) -> pd.DataFrame:
        """
        Predict tax evasion risk for multiple companies
        FR24: Support analysis of multiple companies
        
        Args:
            companies_df: DataFrame containing multiple companies' financial data
            preprocess: Whether to preprocess the data first
        
        Returns:
            DataFrame with original data plus risk assessments
        """
        logger.info(f"Starting batch prediction for {len(companies_df)} companies")
        
        # Step 1: Preprocess data if requested
        if preprocess:
            logger.info("Preprocessing data...")
            companies_df = self.data_processor.process_pipeline(
                companies_df, 
                normalize=False  # Don't normalize for GPT-4 analysis
            )
        
        # Step 2: Convert to list of dictionaries
        companies_data = self.data_processor.prepare_for_gpt_analysis(companies_df)
        
        # Step 3: Analyze all companies
        assessments = self.gpt_analyzer.analyze_batch(companies_data)
        
        # Step 4: Convert assessments to DataFrame
        assessments_df = pd.DataFrame(assessments)
        
        # Step 5: Merge with original data
        result_df = companies_df.merge(
            assessments_df,
            on='company_name',
            how='left',
            suffixes=('_original', '_assessment')
        )
        
        logger.info(f"Batch prediction complete. Analyzed {len(assessments)} companies")
        
        # Log risk distribution
        risk_distribution = result_df['risk_level'].value_counts().to_dict()
        logger.info(f"Risk distribution: {risk_distribution}")
        
        return result_df
    
    def get_high_risk_companies(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter and return only high-risk companies
        Useful for prioritizing audits
        """
        high_risk = results_df[results_df['risk_level'] == 2].copy()
        
        # Sort by confidence score (highest first)
        high_risk = high_risk.sort_values('confidence_score', ascending=False)
        
        logger.info(f"Found {len(high_risk)} high-risk companies")
        
        return high_risk
    
    def get_risk_summary(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for risk assessment results
        FR22: Generate analytical reports
        """
        total_companies = len(results_df)
        
        # Count by risk level
        no_risk = len(results_df[results_df['risk_level'] == 0])
        medium_risk = len(results_df[results_df['risk_level'] == 1])
        high_risk = len(results_df[results_df['risk_level'] == 2])
        failed = len(results_df[results_df['risk_level'] == -1])
        
        # Calculate percentages
        summary = {
            'total_companies': total_companies,
            'risk_distribution': {
                'no_risk': {
                    'count': int(no_risk),
                    'percentage': round((no_risk / total_companies * 100), 2) if total_companies > 0 else 0
                },
                'medium_risk': {
                    'count': int(medium_risk),
                    'percentage': round((medium_risk / total_companies * 100), 2) if total_companies > 0 else 0
                },
                'high_risk': {
                    'count': int(high_risk),
                    'percentage': round((high_risk / total_companies * 100), 2) if total_companies > 0 else 0
                },
                'analysis_failed': {
                    'count': int(failed),
                    'percentage': round((failed / total_companies * 100), 2) if total_companies > 0 else 0
                }
            },
            'average_confidence': round(results_df['confidence_score'].mean(), 3) if 'confidence_score' in results_df.columns else 0,
            'needs_manual_review': int(results_df.get('needs_manual_review', False).sum()) if 'needs_manual_review' in results_df.columns else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def explain_prediction(self, company_data: Dict, assessment: Dict) -> str:
        """
        Generate user-friendly explanation
        FR21: Highlight key financial factors
        """
        try:
            explanation = self.gpt_analyzer.generate_explanation(assessment, company_data)
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate explanation: {str(e)}")
            # Fallback to basic explanation
            return self._generate_basic_explanation(assessment)
    
    def _generate_basic_explanation(self, assessment: Dict) -> str:
        """Generate basic explanation without GPT-4"""
        risk_category = assessment.get('risk_category', 'Unknown')
        confidence = assessment.get('confidence_score', 0)
        key_findings = assessment.get('key_findings', [])
        
        explanation = f"This company has been classified as **{risk_category}** "
        explanation += f"with a confidence level of {confidence:.1%}.\n\n"
        
        if key_findings:
            explanation += "**Key Findings:**\n"
            for finding in key_findings:
                explanation += f"- {finding}\n"
        
        return explanation
# Risk Prediction Engine
