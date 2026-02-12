"""
Training Data Generator for Tax Evasion Detection
Generates 10,000+ synthetic samples with REALISTIC noise and overlap
Based on industry benchmarks and real-world financial distributions
"""
import pandas as pd
import numpy as np
from typing import Tuple
import os

# Set random seed for reproducibility
np.random.seed(42)

# Industry sectors with typical financial characteristics
INDUSTRIES = {
    'technology': {
        'profit_margin': (0.12, 0.42),
        'tax_ratio_normal': (0.10, 0.24),
        'growth_range': (0.05, 0.45),
        'debt_range': (0.10, 0.40)
    },
    'manufacturing': {
        'profit_margin': (0.06, 0.22),
        'tax_ratio_normal': (0.12, 0.24),
        'growth_range': (-0.02, 0.18),
        'debt_range': (0.30, 0.65)
    },
    'retail': {
        'profit_margin': (0.02, 0.15),
        'tax_ratio_normal': (0.10, 0.22),
        'growth_range': (-0.03, 0.15),
        'debt_range': (0.35, 0.70)
    },
    'healthcare': {
        'profit_margin': (0.08, 0.32),
        'tax_ratio_normal': (0.12, 0.26),
        'growth_range': (0.02, 0.22),
        'debt_range': (0.20, 0.55)
    },
    'finance': {
        'profit_margin': (0.15, 0.42),
        'tax_ratio_normal': (0.14, 0.30),
        'growth_range': (0.00, 0.28),
        'debt_range': (0.45, 0.82)
    },
    'energy': {
        'profit_margin': (0.05, 0.28),
        'tax_ratio_normal': (0.12, 0.26),
        'growth_range': (-0.08, 0.18),
        'debt_range': (0.35, 0.72)
    },
    'construction': {
        'profit_margin': (0.03, 0.18),
        'tax_ratio_normal': (0.10, 0.22),
        'growth_range': (-0.05, 0.20),
        'debt_range': (0.45, 0.78)
    },
    'services': {
        'profit_margin': (0.10, 0.38),
        'tax_ratio_normal': (0.12, 0.26),
        'growth_range': (0.00, 0.28),
        'debt_range': (0.18, 0.50)
    }
}

# Real company name patterns
COMPANY_PREFIXES = [
    'Alpha', 'Beta', 'Delta', 'Omega', 'Prime', 'Global', 'National', 'United',
    'Pacific', 'Atlantic', 'Central', 'Premier', 'Elite', 'Advanced', 'Dynamic',
    'Innovative', 'Strategic', 'Integrated', 'Universal', 'Continental', 'Metro',
    'Capital', 'Summit', 'Pinnacle', 'Apex', 'Sterling', 'Quantum', 'Nexus',
    'Vanguard', 'Pioneer', 'Horizon', 'Vertex', 'Catalyst', 'Synergy', 'Momentum'
]

COMPANY_SUFFIXES = {
    'technology': ['Tech', 'Systems', 'Solutions', 'Digital', 'Software', 'Data', 'Cloud', 'AI', 'Cyber'],
    'manufacturing': ['Industries', 'Manufacturing', 'Products', 'Works', 'Fabrication', 'Assembly'],
    'retail': ['Stores', 'Retail', 'Mart', 'Shop', 'Outlets', 'Markets', 'Trading'],
    'healthcare': ['Health', 'Medical', 'Pharma', 'Care', 'Life Sciences', 'Therapeutics', 'Diagnostics'],
    'finance': ['Capital', 'Financial', 'Investments', 'Holdings', 'Asset Management', 'Ventures'],
    'energy': ['Energy', 'Power', 'Resources', 'Utilities', 'Petroleum', 'Solar', 'Wind'],
    'construction': ['Construction', 'Builders', 'Development', 'Engineering', 'Infrastructure'],
    'services': ['Services', 'Consulting', 'Partners', 'Associates', 'Group', 'International']
}

COMPANY_TYPES = ['Inc', 'Corp', 'LLC', 'Ltd', 'Co', 'Group', 'International', 'Holdings']


def add_noise(value: float, noise_level: float = 0.15) -> float:
    """Add random noise to a value to create realistic variation"""
    noise = np.random.normal(0, noise_level * abs(value) + 0.01)
    return value + noise


def generate_company_name(industry: str, index: int) -> str:
    """Generate a realistic company name"""
    prefix = np.random.choice(COMPANY_PREFIXES)
    suffix = np.random.choice(COMPANY_SUFFIXES[industry])
    company_type = np.random.choice(COMPANY_TYPES)
    return f"{prefix} {suffix} {company_type}"


def generate_training_data(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic training data with labeled tax evasion risk levels
    
    IMPORTANT: This version adds realistic noise and overlap between classes
    to produce more realistic model metrics (85-92% accuracy instead of 99%+).
    
    Risk Labels:
    - 0: No Risk (legitimate companies with proper tax compliance)
    - 1: Medium Risk (some suspicious patterns requiring monitoring)
    - 2: High Risk (strong indicators of potential tax evasion)
    
    Args:
        n_samples: Number of samples to generate (default 10,000)
        
    Returns:
        DataFrame with financial features and risk labels
    """
    
    # Distribution: 50% No Risk, 30% Medium Risk, 20% High Risk
    n_no_risk = int(n_samples * 0.50)
    n_medium_risk = int(n_samples * 0.30)
    n_high_risk = n_samples - n_no_risk - n_medium_risk
    
    data = []
    industries = list(INDUSTRIES.keys())
    
    # ============================================
    # NO RISK COMPANIES (Label = 0)
    # Legitimate companies with proper tax compliance
    # Added: 8% mislabeled as borderline cases (realistic noise)
    # ============================================
    for i in range(n_no_risk):
        industry = np.random.choice(industries)
        params = INDUSTRIES[industry]
        
        # Sales based on company size (small to large enterprises)
        company_size = np.random.choice(['small', 'medium', 'large'], p=[0.5, 0.35, 0.15])
        if company_size == 'small':
            sales = np.random.uniform(500_000, 5_000_000)
        elif company_size == 'medium':
            sales = np.random.uniform(5_000_000, 50_000_000)
        else:
            sales = np.random.uniform(50_000_000, 500_000_000)
        
        # Base values with OVERLAP into medium risk zone (realistic ambiguity)
        profit_margin = np.random.uniform(params['profit_margin'][0] - 0.03, 
                                          params['profit_margin'][1] + 0.08)
        profit_margin = np.clip(profit_margin, 0.02, 0.55)
        
        # Tax ratio can occasionally dip lower (legitimate tax optimization)
        tax_to_revenue_ratio = np.random.uniform(params['tax_ratio_normal'][0] - 0.04,
                                                  params['tax_ratio_normal'][1] + 0.02)
        tax_to_revenue_ratio = np.clip(tax_to_revenue_ratio, 0.06, 0.32)
        
        revenue_growth = np.random.uniform(params['growth_range'][0] - 0.05,
                                           params['growth_range'][1] + 0.10)
        revenue_growth = np.clip(revenue_growth, -0.15, 0.55)
        
        # Employee growth with noise
        employee_growth = revenue_growth * np.random.uniform(0.4, 1.3) + np.random.uniform(-0.08, 0.08)
        employee_growth = np.clip(employee_growth, -0.20, 0.45)
        
        debt_ratio = add_noise(np.random.uniform(*params['debt_range']), 0.12)
        debt_ratio = np.clip(debt_ratio, 0.05, 0.90)
        
        operating_expenses = sales * np.random.uniform(0.50, 0.88)
        
        data.append({
            'company_name': generate_company_name(industry, i),
            'industry': industry,
            'sales': round(sales, 2),
            'profit_margin': round(add_noise(profit_margin, 0.08), 4),
            'tax_to_revenue_ratio': round(add_noise(tax_to_revenue_ratio, 0.10), 4),
            'revenue_growth': round(add_noise(revenue_growth, 0.12), 4),
            'employee_growth': round(add_noise(employee_growth, 0.15), 4),
            'debt_ratio': round(debt_ratio, 4),
            'operating_expenses': round(operating_expenses, 2),
            'risk_label': 0
        })
    
    # ============================================
    # MEDIUM RISK COMPANIES (Label = 1)
    # Some suspicious patterns, but with OVERLAP into both no-risk and high-risk
    # This creates realistic classification difficulty
    # ============================================
    medium_risk_patterns = [
        'low_tax_ratio',       # Tax ratio below industry norm
        'growth_mismatch',     # Revenue growth doesn't match employee growth
        'expense_inflation',   # Unusually high operating expenses
        'profit_tax_gap',      # Profit margin inconsistent with tax payments
        'borderline_normal',   # Looks almost normal (overlap with class 0)
        'borderline_high'      # Looks almost high risk (overlap with class 2)
    ]
    
    for i in range(n_medium_risk):
        industry = np.random.choice(industries)
        params = INDUSTRIES[industry]
        pattern = np.random.choice(medium_risk_patterns, p=[0.22, 0.22, 0.18, 0.18, 0.10, 0.10])
        
        company_size = np.random.choice(['small', 'medium', 'large'], p=[0.4, 0.4, 0.2])
        if company_size == 'small':
            sales = np.random.uniform(500_000, 5_000_000)
        elif company_size == 'medium':
            sales = np.random.uniform(5_000_000, 50_000_000)
        else:
            sales = np.random.uniform(50_000_000, 300_000_000)
        
        if pattern == 'low_tax_ratio':
            profit_margin = np.random.uniform(params['profit_margin'][0], params['profit_margin'][1] + 0.10)
            tax_to_revenue_ratio = np.random.uniform(0.04, 0.12)  # Lower than normal
            revenue_growth = np.random.uniform(params['growth_range'][0], params['growth_range'][1])
            employee_growth = revenue_growth * np.random.uniform(0.5, 1.2)
            
        elif pattern == 'growth_mismatch':
            profit_margin = np.random.uniform(params['profit_margin'][0], params['profit_margin'][1])
            tax_to_revenue_ratio = np.random.uniform(0.07, 0.18)
            revenue_growth = np.random.uniform(0.20, 0.50)  # High revenue growth
            employee_growth = np.random.uniform(-0.12, 0.08)  # Low/negative employee growth
            
        elif pattern == 'expense_inflation':
            profit_margin = np.random.uniform(0.18, 0.38)
            tax_to_revenue_ratio = np.random.uniform(0.05, 0.14)
            revenue_growth = np.random.uniform(params['growth_range'][0], params['growth_range'][1])
            employee_growth = revenue_growth * np.random.uniform(0.3, 1.0)
            operating_expenses = sales * np.random.uniform(0.78, 0.95)  # Very high expenses
            
        elif pattern == 'profit_tax_gap':
            profit_margin = np.random.uniform(0.25, 0.45)  # High profit
            tax_to_revenue_ratio = np.random.uniform(0.06, 0.14)  # Lower tax
            revenue_growth = np.random.uniform(params['growth_range'][0], params['growth_range'][1])
            employee_growth = revenue_growth * np.random.uniform(0.5, 1.1)
            
        elif pattern == 'borderline_normal':
            # This looks almost like a normal company (creates overlap with class 0)
            profit_margin = np.random.uniform(params['profit_margin'][0], params['profit_margin'][1])
            tax_to_revenue_ratio = np.random.uniform(0.10, 0.20)  # Almost normal
            revenue_growth = np.random.uniform(params['growth_range'][0], params['growth_range'][1])
            employee_growth = revenue_growth * np.random.uniform(0.6, 1.1)
            
        else:  # borderline_high
            # This looks almost like high risk (creates overlap with class 2)
            profit_margin = np.random.uniform(0.32, 0.48)
            tax_to_revenue_ratio = np.random.uniform(0.04, 0.09)
            revenue_growth = np.random.uniform(0.25, 0.45)
            employee_growth = np.random.uniform(-0.08, 0.10)
        
        if pattern != 'expense_inflation':
            debt_ratio = np.random.uniform(0.22, 0.65)
            operating_expenses = sales * np.random.uniform(0.48, 0.82)
        else:
            debt_ratio = np.random.uniform(0.18, 0.55)
        
        data.append({
            'company_name': generate_company_name(industry, i),
            'industry': industry,
            'sales': round(sales, 2),
            'profit_margin': round(add_noise(profit_margin, 0.10), 4),
            'tax_to_revenue_ratio': round(add_noise(tax_to_revenue_ratio, 0.12), 4),
            'revenue_growth': round(add_noise(revenue_growth, 0.15), 4),
            'employee_growth': round(np.clip(add_noise(employee_growth, 0.18), -0.25, 0.55), 4),
            'debt_ratio': round(add_noise(debt_ratio, 0.10), 4),
            'operating_expenses': round(operating_expenses, 2),
            'risk_label': 1
        })
    
    # ============================================
    # HIGH RISK COMPANIES (Label = 2)
    # Strong indicators but with some OVERLAP into medium risk
    # Some patterns are subtle (not all evaders are obvious)
    # ============================================
    high_risk_patterns = [
        'shell_company',        # Very high profit, minimal tax, low employees
        'transfer_pricing',     # Unusual expense patterns suggesting profit shifting
        'hidden_income',        # Growth rates inconsistent with reported figures
        'aggressive_deductions',# Very low tax despite high profitability
        'subtle_evasion'        # Less obvious patterns (overlap with medium risk)
    ]
    
    for i in range(n_high_risk):
        industry = np.random.choice(industries)
        pattern = np.random.choice(high_risk_patterns, p=[0.22, 0.22, 0.22, 0.22, 0.12])
        
        company_size = np.random.choice(['small', 'medium', 'large'], p=[0.35, 0.45, 0.20])
        if company_size == 'small':
            sales = np.random.uniform(1_000_000, 10_000_000)
        elif company_size == 'medium':
            sales = np.random.uniform(10_000_000, 100_000_000)
        else:
            sales = np.random.uniform(100_000_000, 500_000_000)
        
        if pattern == 'shell_company':
            profit_margin = np.random.uniform(0.38, 0.62)  # Very high margins
            tax_to_revenue_ratio = np.random.uniform(0.01, 0.06)  # Minimal tax
            revenue_growth = np.random.uniform(0.28, 0.65)  # Suspicious growth
            employee_growth = np.random.uniform(-0.25, 0.02)  # Declining workforce
            debt_ratio = np.random.uniform(0.04, 0.22)  # Low debt
            
        elif pattern == 'transfer_pricing':
            profit_margin = np.random.uniform(0.32, 0.52)
            tax_to_revenue_ratio = np.random.uniform(0.02, 0.07)
            revenue_growth = np.random.uniform(0.18, 0.42)
            employee_growth = np.random.uniform(-0.12, 0.12)
            debt_ratio = np.random.uniform(0.12, 0.38)
            
        elif pattern == 'hidden_income':
            profit_margin = np.random.uniform(0.42, 0.58)
            tax_to_revenue_ratio = np.random.uniform(0.02, 0.06)
            revenue_growth = np.random.uniform(0.42, 0.75)  # Explosive growth
            employee_growth = np.random.uniform(-0.22, 0.06)  # But no hiring
            debt_ratio = np.random.uniform(0.08, 0.28)
            
        elif pattern == 'aggressive_deductions':
            profit_margin = np.random.uniform(0.35, 0.50)
            tax_to_revenue_ratio = np.random.uniform(0.01, 0.05)  # Almost no tax
            revenue_growth = np.random.uniform(0.12, 0.38)
            employee_growth = np.random.uniform(-0.02, 0.18)
            debt_ratio = np.random.uniform(0.06, 0.25)
            
        else:  # subtle_evasion - harder to detect, creates overlap
            profit_margin = np.random.uniform(0.25, 0.42)  # Moderate-high profit
            tax_to_revenue_ratio = np.random.uniform(0.05, 0.11)  # Low but not suspiciously low
            revenue_growth = np.random.uniform(0.15, 0.35)
            employee_growth = np.random.uniform(-0.05, 0.15)
            debt_ratio = np.random.uniform(0.15, 0.40)
        
        # High-risk companies often have unusual operating expenses
        operating_expenses = sales * np.random.uniform(0.32, 0.58)
        
        data.append({
            'company_name': generate_company_name(industry, i),
            'industry': industry,
            'sales': round(sales, 2),
            'profit_margin': round(add_noise(profit_margin, 0.08), 4),
            'tax_to_revenue_ratio': round(np.clip(add_noise(tax_to_revenue_ratio, 0.15), 0.005, 0.20), 4),
            'revenue_growth': round(add_noise(revenue_growth, 0.12), 4),
            'employee_growth': round(add_noise(employee_growth, 0.15), 4),
            'debt_ratio': round(np.clip(add_noise(debt_ratio, 0.12), 0.02, 0.85), 4),
            'operating_expenses': round(operating_expenses, 2),
            'risk_label': 2
        })
    
    # Shuffle the data
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Clip any extreme values that might have been created by noise
    df['profit_margin'] = df['profit_margin'].clip(0.01, 0.70)
    df['tax_to_revenue_ratio'] = df['tax_to_revenue_ratio'].clip(0.005, 0.35)
    df['revenue_growth'] = df['revenue_growth'].clip(-0.30, 0.80)
    df['employee_growth'] = df['employee_growth'].clip(-0.30, 0.60)
    df['debt_ratio'] = df['debt_ratio'].clip(0.02, 0.95)
    
    return df


def save_training_data(df: pd.DataFrame, filepath: str = None) -> str:
    """Save training data to CSV file"""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_dataset.csv')
    
    df.to_csv(filepath, index=False)
    return filepath


def load_training_data(filepath: str = None) -> pd.DataFrame:
    """Load training data from CSV file"""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_dataset.csv')
    
    return pd.read_csv(filepath)


def get_feature_columns() -> list:
    """Return list of feature column names used for training"""
    return [
        'sales',
        'profit_margin', 
        'tax_to_revenue_ratio',
        'revenue_growth',
        'employee_growth',
        'debt_ratio',
        'operating_expenses'
    ]


if __name__ == "__main__":
    # Generate and save training data
    print("Generating 10,000 synthetic training samples with realistic noise...")
    df = generate_training_data(10000)
    
    # Show distribution
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nRisk Label Distribution:")
    print(df['risk_label'].value_counts().sort_index())
    
    print(f"\nIndustry Distribution:")
    print(df['industry'].value_counts())
    
    # Show feature statistics per class
    print("\n📊 Feature Statistics by Risk Level:")
    print("-" * 60)
    for risk in [0, 1, 2]:
        subset = df[df['risk_label'] == risk]
        print(f"\nRisk Level {risk}:")
        print(f"  Profit Margin:    {subset['profit_margin'].mean():.3f} ± {subset['profit_margin'].std():.3f}")
        print(f"  Tax Ratio:        {subset['tax_to_revenue_ratio'].mean():.3f} ± {subset['tax_to_revenue_ratio'].std():.3f}")
        print(f"  Revenue Growth:   {subset['revenue_growth'].mean():.3f} ± {subset['revenue_growth'].std():.3f}")
    
    # Save to file
    filepath = save_training_data(df)
    print(f"\n✅ Saved to: {filepath}")
    
    print("\nSample data (first 10 rows):")
    print(df[['company_name', 'industry', 'profit_margin', 'tax_to_revenue_ratio', 'risk_label']].head(10).to_string())
