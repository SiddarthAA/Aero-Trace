"""
LLR-specific prompts for reasoning and classification
"""

def get_llr_reasoning_prompt(llr_text: str, hlr_text: str, similarity: float) -> str:
    """
    Generate reasoning prompt for LLR-HLR relationship
    
    This analyzes from LLR perspective:
    - Does this LLR support this HLR?
    - What role does it play?
    - Is it necessary or extra behavior?
    
    Args:
        llr_text: LLR embedding text
        hlr_text: HLR embedding text
        similarity: Similarity score
        
    Returns:
        Formatted prompt
    """
    prompt = f"""Analyze the relationship between this Low-Level Requirement (LLR) and a High-Level Requirement (HLR).

**LLR (Low-Level Requirement):**
{llr_text}

**HLR (High-Level Requirement):**
{hlr_text}

**Semantic Similarity Score:** {similarity:.3f}

Answer these questions:

1. **Role Alignment**: If this LLR supports the HLR, what role does it play?
   - Detection/Sensing
   - Action/Response
   - Timing/Performance
   - Safety Constraint
   - Error Handling
   - Data Management
   - Optimization/Enhancement

2. **Does this LLR support the HLR's intent?**
   Consider: Does it help achieve what the HLR requires?
   Answer: YES or NO

3. **Is this LLR necessary for the HLR?**
   Would the HLR be incomplete or violated without this LLR?
   Answer: YES or NO

4. **Does this LLR extend beyond the HLR's scope?**
   Does it add behavior NOT demanded or implied by the HLR?
   Answer: YES or NO

Provide your analysis in this format:
ROLE: <role name>
SUPPORTS_HLR: <YES/NO>
IS_NECESSARY: <YES/NO>
EXTENDS_BEYOND: <YES/NO>
REASONING: <brief explanation>
"""
    return prompt


def get_llr_classification_prompt(llr: dict, hlr_candidates: list) -> str:
    """
    Generate classification prompt for an LLR based on its HLR relationships
    
    Args:
        llr: LLR dictionary
        hlr_candidates: List of HLR relationship analysis results
        
    Returns:
        Formatted prompt
    """
    llr_text = llr.get('embedding_text', '')
    
    # Build candidate summary
    candidate_summary = []
    for i, candidate in enumerate(hlr_candidates, 1):
        candidate_summary.append(f"""
HLR Candidate {i}:
  - HLR ID: {candidate['hlr_id']}
  - Similarity: {candidate['similarity']:.3f}
  - Weight: {candidate['weight']:.3f}
  - Supports HLR: {candidate['supports_hlr']}
  - Is Necessary: {candidate['is_necessary']}
  - Extends Beyond: {candidate['extends_beyond']}
        """)
    
    candidates_text = "\n".join(candidate_summary)
    
    prompt = f"""You are a requirements traceability expert. Classify this Low-Level Requirement (LLR) based on its relationship to High-Level Requirements (HLRs).

**LLR to Classify:**
{llr_text}

**Analysis of HLR Relationships:**
{candidates_text}

**Classification Categories:**

1. **COMPLIANT**: 
   - LLR strongly maps to exactly ONE HLR (weight > 0.75)
   - LLR is necessary for that HLR
   - Clear justification exists

2. **AMBIGUOUS**:
   - LLR weakly maps to MULTIPLE HLRs (weight 0.45-0.70)
   - OR LLR is too generic/vague
   - OR unclear which HLR it primarily supports

3. **UNJUSTIFIED**:
   - LLR maps to an HLR but adds behavior beyond what HLR demands
   - Extends beyond system intent
   - Fails necessity test but somewhat related

4. **ORPHAN**:
   - LLR does NOT meaningfully map to ANY HLR (all weights < 0.40)
   - OR introduces behavior not requested at system level
   - No clear system justification

**Your Task:**
Classify this LLR and explain your reasoning.

Consider:
- How many HLRs does it strongly relate to?
- Is it necessary for any HLR, or is it extra behavior?
- Is the LLR scope clear or vague?

Provide classification, confidence (0.0-1.0), and reasoning.
"""
    return prompt


def get_llr_structured_output_schema() -> str:
    """
    Get JSON schema description for LLR classification output
    
    Returns:
        Schema description
    """
    schema = """
Return a JSON object with this exact structure:

{
  "classification": "COMPLIANT" | "AMBIGUOUS" | "UNJUSTIFIED" | "ORPHAN",
  "confidence_score": <float 0.0-1.0>,
  "reasoning": {
    "summary": "<one-sentence classification rationale>",
    "key_findings": [
      "<finding 1>",
      "<finding 2>"
    ],
    "primary_hlr": "<HLR ID if exists, else null>",
    "justification_strength": "STRONG" | "MEDIUM" | "WEAK" | "NONE"
  }
}

Example:
{
  "classification": "COMPLIANT",
  "confidence_score": 0.87,
  "reasoning": {
    "summary": "LLR directly implements detection mechanism required by HLR01",
    "key_findings": [
      "Strong mapping to HLR01 (weight: 0.89)",
      "Provides necessary validation behavior",
      "Clear, specific requirement scope"
    ],
    "primary_hlr": "HLR01",
    "justification_strength": "STRONG"
  }
}
"""
    return schema


def parse_llr_reasoning_response(response: str) -> dict:
    """
    Parse LLR reasoning response from LLM
    
    Args:
        response: Raw LLM response
        
    Returns:
        Parsed reasoning dictionary
    """
    reasoning = {
        'role': 'UNKNOWN',
        'supports_hlr': False,
        'is_necessary': False,
        'extends_beyond': False,
        'explanation': ''
    }
    
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        
        if line.startswith('ROLE:'):
            reasoning['role'] = line.split(':', 1)[1].strip()
        
        elif line.startswith('SUPPORTS_HLR:'):
            value = line.split(':', 1)[1].strip().upper()
            reasoning['supports_hlr'] = value == 'YES'
        
        elif line.startswith('IS_NECESSARY:'):
            value = line.split(':', 1)[1].strip().upper()
            reasoning['is_necessary'] = value == 'YES'
        
        elif line.startswith('EXTENDS_BEYOND:'):
            value = line.split(':', 1)[1].strip().upper()
            reasoning['extends_beyond'] = value == 'YES'
        
        elif line.startswith('REASONING:'):
            reasoning['explanation'] = line.split(':', 1)[1].strip()
    
    return reasoning
