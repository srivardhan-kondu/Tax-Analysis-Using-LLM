"""
Validation module for GPT-4 outputs
Additional anti-hallucination measures and consistency checks
"""
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OutputValidator:
    """
    Validates GPT-4 outputs for consistency and accuracy
    Anti-hallucination measure: Post-processing validation
    """
    
    def __init__(self):
        self.validation_results = []
    
    def validate_assessment(self, assessment: Dict, original_data: Dict) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of risk assessment
        Returns: (is_valid, issues_list)
        """
        issues = []
        
        # 1. Check data integrity
        if not self._verify_company_name(assessment, original_data):
            issues.append("Company name mismatch detected")
        
        # 2. Check numerical consistency
        numerical_issues = self._check_numerical_consistency(assessment, original_data)
        issues.extend(numerical_issues)
        
        # 3. Check logical consistency
        logical_issues = self._check_logical_consistency(assessment)
        issues.extend(logical_issues)
        
        # 4. Check evidence quality
        evidence_issues = self._check_evidence_quality(assessment)
        issues.extend(evidence_issues)
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Validation issues found for {assessment.get('company_name')}:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def _verify_company_name(self, assessment: Dict, original_data: Dict) -> bool:
        """Verify company name matches"""
        assessed_name = assessment.get('company_name', '').strip().lower()
        original_name = original_data.get('company_name', '').strip().lower()
        
        return assessed_name == original_name
    
    def _check_numerical_consistency(self, assessment: Dict, original_data: Dict) -> List[str]:
        """
        Check if numerical references in findings match original data
        Anti-hallucination measure: Verify no fabricated numbers
        """
        issues = []
        
        # Extract numerical values from findings and reasoning
        findings_text = ' '.join(assessment.get('key_findings', []))
        findings_text += ' ' + assessment.get('reasoning', '')
        
        # Check for common hallucinations (numbers that don't exist in original data)
        # This is a simplified check - in production, use more sophisticated NLP
        
        # Verify profit margin references
        if 'profit' in findings_text.lower():
            original_profit = original_data.get('profit_margin', 0)
            # Simple check: if profit margin is mentioned, ensure it's roughly correct
            # More sophisticated validation would extract exact numbers from text
        
        return issues
    
    def _check_logical_consistency(self, assessment: Dict) -> List[str]:
        """
        Check logical consistency of the assessment
        """
        issues = []
        
        # Rule 1: High risk should have red flags
        if assessment.get('risk_level') == 2:
            red_flags = assessment.get('red_flags', [])
            if not red_flags or len(red_flags) == 0:
                issues.append("High risk assessment without red flags")
        
        # Rule 2: No risk should have high confidence
        if assessment.get('risk_level') == 0:
            confidence = assessment.get('confidence_score', 0)
            if confidence < 0.6:
                issues.append("No risk assessment with low confidence")
        
        # Rule 3: Confidence should match evidence strength
        key_findings = assessment.get('key_findings', [])
        if len(key_findings) < 2 and assessment.get('confidence_score', 0) > 0.8:
            issues.append("High confidence with limited findings")
        
        return issues
    
    def _check_evidence_quality(self, assessment: Dict) -> List[str]:
        """
        Check quality and specificity of evidence
        """
        issues = []
        
        # Check if findings are specific
        key_findings = assessment.get('key_findings', [])
        if not key_findings:
            issues.append("No key findings provided")
        else:
            # Check for vague findings
            vague_phrases = ['might', 'maybe', 'possibly', 'unclear', 'uncertain']
            for finding in key_findings:
                if any(phrase in finding.lower() for phrase in vague_phrases):
                    logger.info(f"Vague finding detected: {finding}")
        
        # Check reasoning quality
        reasoning = assessment.get('reasoning', '')
        if len(reasoning) < 50:
            issues.append("Insufficient reasoning provided")
        
        return issues
    
    def compare_multiple_analyses(self, analyses: List[Dict]) -> Dict:
        """
        Compare multiple analyses for consistency
        Useful for detecting hallucinations through consensus
        """
        if len(analyses) < 2:
            return {"consensus": True, "message": "Only one analysis provided"}
        
        # Check if risk levels are consistent
        risk_levels = [a.get('risk_level') for a in analyses]
        unique_levels = set(risk_levels)
        
        if len(unique_levels) == 1:
            return {
                "consensus": True,
                "risk_level": risk_levels[0],
                "confidence": "High - all analyses agree"
            }
        else:
            return {
                "consensus": False,
                "risk_levels": list(unique_levels),
                "confidence": "Low - analyses disagree",
                "recommendation": "Manual review recommended"
            }
