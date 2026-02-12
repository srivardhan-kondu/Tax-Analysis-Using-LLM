"""
GPT-4 Analyzer for Tax Evasion Detection
Integrates OpenAI GPT-4 with anti-hallucination measures
"""
import json
import logging
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import config
from ai.prompts import (
    SYSTEM_PROMPT,
    create_analysis_prompt,
    BATCH_ANALYSIS_PROMPT_TEMPLATE,
    VALIDATION_PROMPT_TEMPLATE,
    EXPLANATION_PROMPT_TEMPLATE
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPT4Analyzer:
    """
    GPT-4 based tax evasion risk analyzer with anti-hallucination measures
    Implements intelligent analysis as replacement for ML/DL models
    """
    
    def __init__(self):
        """Initialize GPT-4 analyzer with API client"""
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
        self.temperature = config.GPT_TEMPERATURE  # 0.0 for deterministic output
        self.max_tokens = config.GPT_MAX_TOKENS
        
        logger.info(f"Initialized GPT-4 Analyzer with model: {self.model}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _make_api_call(self, messages: List[Dict], temperature: Optional[float] = None) -> str:
        """
        Make API call to GPT-4 with retry logic
        Anti-hallucination measure: Use low temperature for consistency
        """
        try:
            temp = temperature if temperature is not None else self.temperature
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            content = response.choices[0].message.content
            logger.info(f"API call successful. Tokens used: {response.usage.total_tokens}")
            
            return content
        
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise
    
    def analyze_company(self, company_data: Dict) -> Dict:
        """
        Analyze a single company for tax evasion risk
        FR16: Analyze new financial data using trained models
        FR17: Predict tax evasion risk level
        FR18: Classify into risk categories (0, 1, 2)
        FR19: Generate confidence score
        """
        logger.info(f"Analyzing company: {company_data.get('company_name', 'Unknown')}")
        
        # Step 1: Create analysis prompt
        user_prompt = create_analysis_prompt(company_data)
        
        # Step 2: Make API call with structured output
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        response_text = self._make_api_call(messages)
        
        # Step 3: Parse JSON response
        try:
            assessment = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response was: {response_text}")
            raise ValueError("GPT-4 returned invalid JSON")
        
        # Step 4: Validate the assessment (anti-hallucination measure)
        assessment = self._validate_assessment(assessment, company_data)
        
        # Step 5: Apply confidence threshold
        if assessment['confidence_score'] < config.GPT_CONFIDENCE_THRESHOLD:
            logger.warning(
                f"Low confidence score ({assessment['confidence_score']:.2f}) "
                f"for {company_data.get('company_name')}"
            )
            assessment['needs_manual_review'] = True
        else:
            assessment['needs_manual_review'] = False
        
        return assessment
    
    def analyze_batch(self, companies_data: List[Dict], batch_size: int = 5) -> List[Dict]:
        """
        Analyze multiple companies
        FR24: Support analysis of multiple companies
        """
        logger.info(f"Analyzing batch of {len(companies_data)} companies")
        
        results = []
        
        # Process in smaller batches to avoid context limits
        for i in range(0, len(companies_data), batch_size):
            batch = companies_data[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} companies)")
            
            # Analyze each company individually for better accuracy
            for company_data in batch:
                try:
                    assessment = self.analyze_company(company_data)
                    results.append(assessment)
                except Exception as e:
                    logger.error(f"Error analyzing {company_data.get('company_name')}: {str(e)}")
                    # Add error result
                    results.append({
                        'company_name': company_data.get('company_name', 'Unknown'),
                        'risk_level': -1,
                        'risk_category': 'Analysis Failed',
                        'confidence_score': 0.0,
                        'error': str(e),
                        'needs_manual_review': True
                    })
        
        return results
    
    def _validate_assessment(self, assessment: Dict, original_data: Dict) -> Dict:
        """
        Validate assessment for consistency and accuracy
        Anti-hallucination measure: Multi-step verification
        """
        # Ensure required fields are present
        required_fields = ['company_name', 'risk_level', 'confidence_score', 
                          'risk_category', 'key_findings', 'reasoning']
        
        for field in required_fields:
            if field not in assessment:
                logger.warning(f"Missing required field in assessment: {field}")
                assessment[field] = self._get_default_value(field, original_data)
        
        # Validate risk level is valid (0, 1, or 2)
        if assessment['risk_level'] not in [0, 1, 2]:
            logger.warning(f"Invalid risk level: {assessment['risk_level']}, defaulting to 1")
            assessment['risk_level'] = 1
            assessment['risk_category'] = 'Medium Risk'
            assessment['confidence_score'] *= 0.5  # Reduce confidence
        
        # Validate confidence score is in range [0, 1]
        if not (0 <= assessment['confidence_score'] <= 1):
            logger.warning(f"Invalid confidence score: {assessment['confidence_score']}")
            assessment['confidence_score'] = max(0, min(1, assessment['confidence_score']))
        
        # Ensure risk category matches risk level
        risk_level_to_category = {
            0: 'No Risk',
            1: 'Medium Risk',
            2: 'High Risk'
        }
        expected_category = risk_level_to_category.get(assessment['risk_level'])
        if assessment['risk_category'] != expected_category:
            logger.warning(
                f"Risk category mismatch. Level: {assessment['risk_level']}, "
                f"Category: {assessment['risk_category']}, Expected: {expected_category}"
            )
            assessment['risk_category'] = expected_category
        
        # Add timestamp
        from datetime import datetime
        assessment['analysis_timestamp'] = datetime.now().isoformat()
        
        # Apply rule-based validation (cross-check with financial rules)
        assessment = self._apply_rule_based_validation(assessment, original_data)
        
        return assessment
    
    def _apply_rule_based_validation(self, assessment: Dict, original_data: Dict) -> Dict:
        """
        Apply rule-based validation to cross-check GPT-4 assessment
        Anti-hallucination measure: Rule-based cross-checking
        """
        rules_applied = []
        confidence_adjustments = []
        
        # Rule 1: Very low tax ratio should trigger high risk
        tax_ratio = original_data.get('tax_to_revenue_ratio', 0)
        profit_margin = original_data.get('profit_margin', 0)
        
        if tax_ratio < config.RISK_THRESHOLDS['tax_to_revenue_ratio_low'] and profit_margin > 0.15:
            if assessment['risk_level'] == 0:
                logger.warning("Rule-based validation: Upgrading risk level due to low tax ratio")
                assessment['risk_level'] = max(assessment['risk_level'], 1)
                assessment['risk_category'] = 'Medium Risk'
                rules_applied.append("Low tax ratio rule triggered")
        
        # Rule 2: High profit margin with low tax suggests evasion
        if profit_margin > config.RISK_THRESHOLDS['profit_margin_high'] and tax_ratio < 0.10:
            if assessment['risk_level'] < 2:
                logger.warning("Rule-based validation: High profit/low tax combination detected")
                assessment['risk_level'] = 2
                assessment['risk_category'] = 'High Risk'
                rules_applied.append("High profit/low tax rule triggered")
        
        # Rule 3: Extreme debt ratio
        debt_ratio = original_data.get('debt_ratio', 0)
        if debt_ratio > config.RISK_THRESHOLDS['debt_ratio_high']:
            rules_applied.append("High debt ratio noted")
        
        # Add rule validation metadata
        if rules_applied:
            assessment['rule_based_validation'] = {
                'rules_triggered': rules_applied,
                'validation_passed': True
            }
        
        return assessment
    
    def _get_default_value(self, field: str, original_data: Dict):
        """Get default value for missing field"""
        defaults = {
            'company_name': original_data.get('company_name', 'Unknown'),
            'risk_level': 1,
            'confidence_score': 0.5,
            'risk_category': 'Medium Risk',
            'key_findings': ['Analysis incomplete - default values used'],
            'reasoning': 'Assessment generated with default values due to missing data',
            'red_flags': []
        }
        return defaults.get(field, None)
    
    def generate_explanation(self, assessment: Dict, company_data: Dict) -> str:
        """
        Generate a user-friendly explanation of the risk assessment
        FR20: Display results in clear and interpretable format
        """
        financial_summary = f"""
- Sales: ${company_data.get('sales', 0):,.2f}
- Profit Margin: {company_data.get('profit_margin', 0):.2%}
- Tax-to-Revenue Ratio: {company_data.get('tax_to_revenue_ratio', 0):.2%}
- Revenue Growth: {company_data.get('revenue_growth', 0):.2%}
"""
        
        assessment_summary = f"""
Risk Level: {assessment['risk_level']} ({assessment['risk_category']})
Confidence: {assessment['confidence_score']:.1%}
Key Findings: {', '.join(assessment.get('key_findings', [])[:3])}
"""
        
        prompt = EXPLANATION_PROMPT_TEMPLATE.format(
            risk_category=assessment['risk_category'],
            company_name=assessment['company_name'],
            risk_level=assessment['risk_level'],
            financial_summary=financial_summary,
            assessment_summary=assessment_summary
        )
        
        messages = [
            {"role": "system", "content": "You are a financial analyst explaining risk assessments to non-technical stakeholders."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            explanation = self._make_api_call(messages, temperature=0.3)
            # Parse JSON if it's wrapped
            try:
                parsed = json.loads(explanation)
                if 'explanation' in parsed:
                    explanation = parsed['explanation']
            except:
                pass  # Not JSON, use as is
            
            return explanation
        except Exception as e:
            logger.error(f"Failed to generate explanation: {str(e)}")
            return assessment.get('reasoning', 'No explanation available')
# AI Module for Tax Evasion Detection
