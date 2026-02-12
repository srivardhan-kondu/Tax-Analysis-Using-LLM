"""
Data processor for Tax Evasion Detection System
Handles data cleaning, preprocessing, and feature engineering
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles all data preprocessing operations for financial data
    Implements FR6, FR7, FR8, FR9
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.required_columns = [
            'company_name',
            'sales',
            'revenue_growth',
            'profit_margin',
            'employee_growth',
            'debt_ratio',
            'operating_expenses',
            'tax_to_revenue_ratio'
        ]
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input dataframe
        FR6: Handle missing values
        FR7: Remove duplicates
        """
        logger.info(f"Initial data shape: {df.shape}")
        
        # Remove duplicates (FR7)
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate records")
        
        # Handle missing values (FR6)
        logger.info(f"Missing values before cleaning:\n{df.isnull().sum()}")
        
        # For numerical columns, fill with median
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if df[col].isnull().any():
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.info(f"Filled {col} missing values with median: {median_value}")
        
        # For categorical columns, fill with mode or 'Unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col].fillna('Unknown', inplace=True)
        
        logger.info(f"Data shape after cleaning: {df.shape}")
        return df
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the dataframe has required columns
        """
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        return True
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from existing data
        FR9: Apply feature transformation techniques
        """
        logger.info("Applying feature engineering...")
        
        # Financial health indicators
        if 'sales' in df.columns and 'operating_expenses' in df.columns:
            df['expense_to_sales_ratio'] = df['operating_expenses'] / (df['sales'] + 1e-6)
        
        # Tax efficiency indicator
        if 'profit_margin' in df.columns and 'tax_to_revenue_ratio' in df.columns:
            df['tax_efficiency'] = df['tax_to_revenue_ratio'] / (df['profit_margin'] + 1e-6)
        
        # Growth consistency check
        if 'revenue_growth' in df.columns and 'employee_growth' in df.columns:
            df['growth_consistency'] = abs(df['revenue_growth'] - df['employee_growth'])
        
        # Profit to tax ratio (high values could indicate evasion)
        if 'profit_margin' in df.columns and 'tax_to_revenue_ratio' in df.columns:
            df['profit_to_tax_ratio'] = df['profit_margin'] / (df['tax_to_revenue_ratio'] + 1e-6)
        
        # Debt burden indicator
        if 'debt_ratio' in df.columns and 'profit_margin' in df.columns:
            df['debt_to_profit'] = df['debt_ratio'] / (df['profit_margin'] + 1e-6)
        
        logger.info(f"Created {len(df.columns) - len(self.required_columns)} new features")
        return df
    
    def normalize_data(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features
        FR8: Normalize and scale numerical financial attributes
        """
        # Preserve non-numerical columns
        non_numerical = df.select_dtypes(include=['object']).columns.tolist()
        numerical = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical:
            logger.warning("No numerical columns to normalize")
            return df
        
        logger.info(f"Normalizing {len(numerical)} numerical columns")
        
        # Create a copy to avoid modifying original
        df_normalized = df.copy()
        
        if fit:
            # Fit and transform
            df_normalized[numerical] = self.scaler.fit_transform(df[numerical])
        else:
            # Only transform (use previously fitted scaler)
            df_normalized[numerical] = self.scaler.transform(df[numerical])
        
        return df_normalized
    
    def process_pipeline(self, df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # Step 1: Validate schema
        if not self.validate_schema(df):
            # Try to map columns if possible
            df = self._attempt_column_mapping(df)
            if not self.validate_schema(df):
                raise ValueError("Data does not match required schema")
        
        # Step 2: Clean data
        df = self.clean_data(df)
        
        # Step 3: Feature engineering
        df = self.feature_engineering(df)
        
        # Step 4: Normalize if requested
        if normalize:
            df = self.normalize_data(df, fit=True)
        
        logger.info("Preprocessing pipeline completed successfully")
        return df
    
    def _attempt_column_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attempt to map common column name variations
        """
        column_mappings = {
            'company': 'company_name',
            'name': 'company_name',
            'revenue': 'sales',
            'total_sales': 'sales',
            'rev_growth': 'revenue_growth',
            'profit': 'profit_margin',
            'emp_growth': 'employee_growth',
            'debt': 'debt_ratio',
            'expenses': 'operating_expenses',
            'opex': 'operating_expenses',
            'tax_ratio': 'tax_to_revenue_ratio',
            'tax_rate': 'tax_to_revenue_ratio',
        }
        
        df_renamed = df.copy()
        for old_name, new_name in column_mappings.items():
            if old_name in df_renamed.columns:
                df_renamed.rename(columns={old_name: new_name}, inplace=True)
                logger.info(f"Mapped column '{old_name}' to '{new_name}'")
        
        return df_renamed
    
    def prepare_for_gpt_analysis(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert dataframe to format suitable for GPT-4 analysis
        Returns list of dictionaries, each representing a company
        """
        # Select relevant columns for analysis
        analysis_columns = [col for col in df.columns if col != 'company_name']
        
        companies_data = []
        for _, row in df.iterrows():
            company_dict = row.to_dict()
            
            # Round numerical values for clarity
            for key, value in company_dict.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    company_dict[key] = round(value, 4)
            
            companies_data.append(company_dict)
        
        return companies_data


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load CSV file and return dataframe
    FR3: Allow users to upload company financial data in CSV format
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded CSV: {file_path}")
        logger.info(f"Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        raise


def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Save processed data to CSV
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise
# Data Processing Module
