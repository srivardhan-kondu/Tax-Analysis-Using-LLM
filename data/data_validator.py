"""
Data validator for Tax Evasion Detection System
Validates input data for correctness and range checks
"""
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates financial data inputs
    Implements data quality checks and range validation
    """
    
    # Define valid ranges for financial indicators
    VALID_RANGES = {
        'sales': (0, float('inf')),  # Must be positive
        'revenue_growth': (-1.0, 5.0),  # -100% to 500%
        'profit_margin': (-1.0, 1.0),  # -100% to 100%
        'employee_growth': (-1.0, 5.0),  # -100% to 500%
        'debt_ratio': (0, 10.0),  # 0 to 1000%
        'operating_expenses': (0, float('inf')),  # Must be positive
        'tax_to_revenue_ratio': (0, 1.0),  # 0% to 100%
    }
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        Validate entire dataframe
        Returns: (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check if dataframe is empty
        if df.empty:
            self.validation_errors.append("DataFrame is empty")
            return False, self.validation_errors, self.validation_warnings
        
        # Validate each row
        for idx, row in df.iterrows():
            self._validate_row(row, idx)
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings
    
    def validate_single_entry(self, data: Dict) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single company entry (for manual input)
        FR5: Validate manually entered data
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        # Check required fields
        required_fields = ['company_name', 'sales', 'revenue_growth', 'profit_margin',
                          'employee_growth', 'debt_ratio', 'operating_expenses',
                          'tax_to_revenue_ratio']
        
        for field in required_fields:
            if field not in data or data[field] is None:
                self.validation_errors.append(f"Missing required field: {field}")
        
        if self.validation_errors:
            return False, self.validation_errors, self.validation_warnings
        
        # Validate ranges
        self._validate_data_dict(data, 0)
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings
    
    def _validate_row(self, row: pd.Series, idx: int):
        """
        Validate a single row of the dataframe
        """
        for column, (min_val, max_val) in self.VALID_RANGES.items():
            if column in row:
                value = row[column]
                
                # Skip NaN values (they should be handled in preprocessing)
                if pd.isna(value):
                    continue
                
                # Check range
                if not (min_val <= value <= max_val):
                    self.validation_errors.append(
                        f"Row {idx}, {column}: value {value} is outside valid range "
                        f"[{min_val}, {max_val}]"
                    )
        
        # Business logic validations
        self._validate_business_logic(row, idx)
    
    def _validate_data_dict(self, data: Dict, idx: int):
        """
        Validate a dictionary of data
        """
        for column, (min_val, max_val) in self.VALID_RANGES.items():
            if column in data:
                value = data[column]
                
                # Check if numeric
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    self.validation_errors.append(
                        f"{column}: value '{value}' is not numeric"
                    )
                    continue
                
                # Check range
                if not (min_val <= value <= max_val):
                    self.validation_errors.append(
                        f"{column}: value {value} is outside valid range "
                        f"[{min_val}, {max_val}]"
                    )
        
        # Business logic validations
        self._validate_business_logic_dict(data, idx)
    
    def _validate_business_logic(self, row: pd.Series, idx: int):
        """
        Validate business logic rules
        """
        # Check if tax ratio is suspiciously low given high profit margin
        if 'profit_margin' in row and 'tax_to_revenue_ratio' in row:
            if not pd.isna(row['profit_margin']) and not pd.isna(row['tax_to_revenue_ratio']):
                if row['profit_margin'] > 0.2 and row['tax_to_revenue_ratio'] < 0.03:
                    self.validation_warnings.append(
                        f"Row {idx}: High profit margin ({row['profit_margin']:.2%}) "
                        f"with low tax ratio ({row['tax_to_revenue_ratio']:.2%}) "
                        f"- potential red flag"
                    )
        
        # Check if operating expenses exceed sales
        if 'sales' in row and 'operating_expenses' in row:
            if not pd.isna(row['sales']) and not pd.isna(row['operating_expenses']):
                if row['operating_expenses'] > row['sales'] * 1.5:
                    self.validation_warnings.append(
                        f"Row {idx}: Operating expenses significantly exceed sales"
                    )
        
        # Check for unusual debt ratio
        if 'debt_ratio' in row:
            if not pd.isna(row['debt_ratio']):
                if row['debt_ratio'] > 1.0:
                    self.validation_warnings.append(
                        f"Row {idx}: Very high debt ratio ({row['debt_ratio']:.2f})"
                    )
    
    def _validate_business_logic_dict(self, data: Dict, idx: int):
        """
        Validate business logic rules for dictionary data
        """
        try:
            # Check if tax ratio is suspiciously low given high profit margin
            if 'profit_margin' in data and 'tax_to_revenue_ratio' in data:
                profit_margin = float(data['profit_margin'])
                tax_ratio = float(data['tax_to_revenue_ratio'])
                
                if profit_margin > 0.2 and tax_ratio < 0.03:
                    self.validation_warnings.append(
                        f"High profit margin ({profit_margin:.2%}) with low tax ratio "
                        f"({tax_ratio:.2%}) - potential red flag"
                    )
            
            # Check if operating expenses exceed sales
            if 'sales' in data and 'operating_expenses' in data:
                sales = float(data['sales'])
                expenses = float(data['operating_expenses'])
                
                if expenses > sales * 1.5:
                    self.validation_warnings.append(
                        "Operating expenses significantly exceed sales"
                    )
            
            # Check for unusual debt ratio
            if 'debt_ratio' in data:
                debt_ratio = float(data['debt_ratio'])
                if debt_ratio > 1.0:
                    self.validation_warnings.append(
                        f"Very high debt ratio ({debt_ratio:.2f})"
                    )
        except (ValueError, TypeError) as e:
            logger.error(f"Error in business logic validation: {str(e)}")
    
    def get_validation_summary(self) -> str:
        """
        Get a human-readable validation summary
        """
        summary = []
        
        if self.validation_errors:
            summary.append(f"❌ {len(self.validation_errors)} Errors:")
            for error in self.validation_errors[:10]:  # Show first 10
                summary.append(f"  - {error}")
            if len(self.validation_errors) > 10:
                summary.append(f"  ... and {len(self.validation_errors) - 10} more")
        
        if self.validation_warnings:
            summary.append(f"⚠️  {len(self.validation_warnings)} Warnings:")
            for warning in self.validation_warnings[:10]:  # Show first 10
                summary.append(f"  - {warning}")
            if len(self.validation_warnings) > 10:
                summary.append(f"  ... and {len(self.validation_warnings) - 10} more")
        
        if not self.validation_errors and not self.validation_warnings:
            summary.append("✅ All validations passed")
        
        return "\n".join(summary)
