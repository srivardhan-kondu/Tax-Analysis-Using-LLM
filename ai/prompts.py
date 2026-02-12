"""
Prompt templates for Tax Evasion Detection using GPT-4
Carefully crafted prompts to ensure accurate, hallucination-free analysis
"""

SYSTEM_PROMPT = """You are a financial analysis AI specialized in detecting tax evasion patterns from company financial data.

Your task is to analyze company financial indicators and assess the risk of tax evasion.

CRITICAL RULES:
1. Base your analysis ONLY on the provided financial data
2. DO NOT make up or assume any information not provided
3. Provide specific numerical evidence for all claims
4. Use only the financial indicators given
5. Your output MUST be in valid JSON format
6. Be conservative - when in doubt, assign lower risk levels

RISK LEVELS:
- 0 (No Risk): Normal financial patterns, appropriate tax ratios
- 1 (Medium Risk): Some concerning patterns, requires monitoring  
- 2 (High Risk): Multiple red flags, high probability of tax evasion

CONFIDENCE SCORING GUIDELINES (VERY IMPORTANT - USE VARIED SCORES):
Assign confidence based on these specific criteria:
- 0.95-1.0: Perfect data, crystal clear pattern, no ambiguity
- 0.85-0.94: Strong evidence, consistent patterns across most metrics
- 0.70-0.84: Good evidence but some metrics are borderline or missing
- 0.55-0.69: Mixed signals, pattern unclear, limited data points
- 0.40-0.54: Weak evidence, many ambiguous indicators
- 0.25-0.39: Very uncertain, data quality poor
- Below 0.25: Insufficient data to make assessment

IMPORTANT: Do NOT default to 0.80! Carefully evaluate each case and assign the most accurate confidence.

You must analyze the following key patterns:
- Tax-to-revenue ratio compared to industry norms (typically 15-25%)
- Profit margins vs tax payments (inconsistencies indicate evasion)
- Revenue growth vs employee growth (should correlate reasonably)
- Debt ratio patterns
- Operating expense ratios
"""

ANALYSIS_PROMPT_TEMPLATE = """Analyze the following company financial data for tax evasion risk:

Company Name: {company_name}

Financial Indicators:
- Sales: ${sales:,.2f}
- Revenue Growth: {revenue_growth:.2%}
- Profit Margin: {profit_margin:.2%}
- Employee Growth: {employee_growth:.2%}
- Debt Ratio: {debt_ratio:.2%}
- Operating Expenses: ${operating_expenses:,.2f}
- Tax-to-Revenue Ratio: {tax_to_revenue_ratio:.2%}

Additional Calculated Metrics:
{additional_metrics}

INSTRUCTIONS:
1. Analyze each financial indicator carefully
2. Identify any suspicious patterns or anomalies
3. Compare ratios against typical ranges
4. Look for inconsistencies between metrics
5. Provide evidence-based assessment

You MUST respond with a JSON object in this EXACT format:
{{
  "company_name": "exact company name from input",
  "risk_level": 0 or 1 or 2,
  "confidence_score": 0.0 to 1.0,
  "risk_category": "No Risk" or "Medium Risk" or "High Risk",
  "key_findings": [
    "specific finding 1 with numerical evidence",
    "specific finding 2 with numerical evidence",
    "specific finding 3 with numerical evidence"
  ],
  "red_flags": [
    "specific red flag 1 (if any)",
    "specific red flag 2 (if any)"
  ],
  "reasoning": "detailed paragraph explaining the risk assessment with specific references to the numbers provided",
  "financial_indicators_analysis": {{
    "tax_to_revenue_ratio": "assessment of this metric",
    "profit_margin": "assessment of this metric",
    "revenue_vs_employee_growth": "assessment of this correlation",
    "debt_ratio": "assessment of this metric",
    "operating_expenses": "assessment of this metric"
  }}
}}

RETURN ONLY THE JSON OBJECT, NO OTHER TEXT.
"""

BATCH_ANALYSIS_PROMPT_TEMPLATE = """Analyze the following {count} companies for tax evasion risk patterns.

For EACH company, analyze the financial indicators and identify risk levels.

Companies Data:
{companies_data}

CRITICAL INSTRUCTIONS:
1. Analyze EACH company independently
2. Base assessments ONLY on the data provided
3. DO NOT make assumptions or create data
4. Provide specific numerical evidence
5. Be consistent in your risk criteria across all companies

Return a JSON array with one object per company in this EXACT format:
[
  {{
    "company_name": "name",
    "risk_level": 0 or 1 or 2,
    "confidence_score": 0.0 to 1.0,
    "risk_category": "category",
    "key_findings": ["finding1", "finding2"],
    "red_flags": ["flag1", "flag2"],
    "reasoning": "explanation"
  }},
  ...
]

RETURN ONLY THE JSON ARRAY, NO OTHER TEXT.
"""

VALIDATION_PROMPT_TEMPLATE = """You are a financial data validation expert. Review the following tax evasion risk assessment for logical consistency and accuracy.

Original Financial Data:
{original_data}

Risk Assessment:
{assessment}

VALIDATION TASKS:
1. Verify all numerical references in the assessment match the original data
2. Check if the risk level is justified by the evidence provided
3. Identify any logical inconsistencies
4. Verify the confidence score is appropriate
5. Check if red flags are supported by actual data

Respond with a JSON object:
{{
  "is_valid": true or false,
  "consistency_score": 0.0 to 1.0,
  "issues_found": ["issue1", "issue2", ...],
  "recommendations": "suggestions for the assessment if any issues found"
}}

RETURN ONLY THE JSON OBJECT, NO OTHER TEXT.
"""

EXPLANATION_PROMPT_TEMPLATE = """Generate a clear, non-technical explanation of why this company was classified as {risk_category}.

Company: {company_name}
Risk Level: {risk_level}
Key Financial Data:
{financial_summary}

Risk Assessment Summary:
{assessment_summary}

Provide a 2-3 paragraph explanation that:
1. Explains the risk classification in simple terms
2. Highlights the most important financial indicators
3. Describes what patterns led to this assessment
4. Avoids technical jargon

Keep it factual and based only on the provided data.
"""


def format_additional_metrics(data: dict) -> str:
    """Format additional calculated metrics for the prompt"""
    metrics = []
    
    if 'expense_to_sales_ratio' in data:
        metrics.append(f"- Expense-to-Sales Ratio: {data['expense_to_sales_ratio']:.2%}")
    
    if 'tax_efficiency' in data:
        metrics.append(f"- Tax Efficiency: {data['tax_efficiency']:.4f}")
    
    if 'growth_consistency' in data:
        metrics.append(f"- Growth Consistency Gap: {data['growth_consistency']:.2%}")
    
    if 'profit_to_tax_ratio' in data:
        metrics.append(f"- Profit-to-Tax Ratio: {data['profit_to_tax_ratio']:.2f}")
    
    if 'debt_to_profit' in data:
        metrics.append(f"- Debt-to-Profit: {data['debt_to_profit']:.2f}")
    
    return "\n".join(metrics) if metrics else "No additional metrics calculated"


def create_analysis_prompt(company_data: dict) -> str:
    """Create a complete analysis prompt for a single company"""
    additional_metrics = format_additional_metrics(company_data)
    
    return ANALYSIS_PROMPT_TEMPLATE.format(
        company_name=company_data.get('company_name', 'Unknown'),
        sales=company_data.get('sales', 0),
        revenue_growth=company_data.get('revenue_growth', 0),
        profit_margin=company_data.get('profit_margin', 0),
        employee_growth=company_data.get('employee_growth', 0),
        debt_ratio=company_data.get('debt_ratio', 0),
        operating_expenses=company_data.get('operating_expenses', 0),
        tax_to_revenue_ratio=company_data.get('tax_to_revenue_ratio', 0),
        additional_metrics=additional_metrics
    )
