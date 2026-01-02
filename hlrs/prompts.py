"""
Prompt templates for reasoning and structured output generation
"""


def get_reasoning_prompt(hlr_text: str, llr_text: str, similarity: float) -> str:
    """
    Generate reasoning prompt for a single HLR-LLR pair
    
    Args:
        hlr_text: HLR requirement text
        llr_text: LLR requirement text
        similarity: Semantic similarity score
        
    Returns:
        Formatted prompt
    """
    return f"""You are analyzing the relationship between a high-level requirement (HLR) and a low-level requirement (LLR).

HLR (System-level requirement):
"{hlr_text}"

LLR (Implementation-level requirement):
"{llr_text}"

Semantic similarity score: {similarity:.3f}

Answer the following 4 questions with clear YES/NO followed by brief reasoning:

1. Is this LLR required for the HLR to hold true?
   (Would the HLR's intent be fulfilled without this LLR?)

2. Would the HLR be violated if this LLR did not exist?
   (Is this LLR critical to preventing HLR violation?)

3. Does the LLR constrain unsafe behavior?
   (Does it add safety constraints, limits, or protections?)

4. Does it extend beyond system intent?
   (Does this LLR introduce behavior NOT demanded by the HLR?)

Format your response as:
Q1: YES/NO - [brief reasoning]
Q2: YES/NO - [brief reasoning]
Q3: YES/NO - [brief reasoning]
Q4: YES/NO - [brief reasoning]"""


def get_classification_prompt(hlr: dict, reasoning_results: list[dict]) -> str:
    """
    Generate prompt for final HLR classification
    
    Args:
        hlr: HLR dictionary
        reasoning_results: List of reasoning results for linked LLRs
        
    Returns:
        Formatted prompt
    """
    hlr_text = hlr.get('embedding_text', '')
    
    # Format reasoning results
    llr_summaries = []
    for i, result in enumerate(reasoning_results, 1):
        llr_summaries.append(f"""
LLR {i}: {result['llr_id']} (similarity: {result['similarity']:.3f})
- Required: {result['is_required']}
- Prevents violation: {result['prevents_violation']}
- Constrains unsafe behavior: {result['constrains_unsafe_behavior']}
- Extends beyond intent: {result['extends_beyond_intent']}
- Weight: {result['weight']:.3f}
""")
    
    llr_summary_text = "\n".join(llr_summaries)
    
    return f"""You are classifying a high-level requirement (HLR) based on its decomposition into low-level requirements (LLRs).

HLR: "{hlr_text}"

Linked LLRs and their analysis:
{llr_summary_text}

Metrics:
- Total candidates: {len(reasoning_results)}
- Strong supporters (weight > 0.75): {sum(1 for r in reasoning_results if r['weight'] > 0.75)}
- Medium supporters (weight 0.50-0.75): {sum(1 for r in reasoning_results if 0.50 < r['weight'] <= 0.75)}
- Weak supporters (weight 0.35-0.50): {sum(1 for r in reasoning_results if 0.35 < r['weight'] <= 0.50)}

Classification criteria:
- FULLY_TRACED: ≥3 strong supporters OR (≥2 strong + ≥2 medium). All expected aspects covered.
- PARTIAL_TRACE: ≥1 strong OR ≥2 medium. Some coverage but incomplete or missing critical aspects.
- TRACE_HOLE: <2 supporters total. No meaningful implementation.

Based on the evidence, classify this HLR and provide:
1. Classification (FULLY_TRACED, PARTIAL_TRACE, or TRACE_HOLE)
2. Confidence score (0.0-0.95)
3. Clear explanation of the decision
4. Key aspects that ARE covered by existing LLRs
5. **For PARTIAL_TRACE only:** Specific gaps - what critical aspects, conditions, or behaviors are MISSING from the current LLR set

For PARTIAL_TRACE classifications, be very specific about what's missing. Analyze the HLR carefully and identify:
- Missing functional aspects (e.g., "No LLR defines the actual degraded control law behavior")
- Missing conditions or triggers (e.g., "No LLR specifies the invalid air data detection criteria")
- Missing safety constraints (e.g., "No LLR addresses fault handling during critical flight phases")
- Missing error handling paths
- Missing performance/timing requirements
- Missing integration or interface requirements

Provide concise, factual, and actionable reasoning."""


def get_structured_output_schema() -> str:
    """
    Get the JSON schema description for structured output
    
    Returns:
        Schema description string
    """
    return """
Return a JSON object with this exact structure:
{
  "classification": "FULLY_TRACED" | "PARTIAL_TRACE" | "TRACE_HOLE",
  "confidence_score": 0.0 to 0.95,
  "reasoning": {
    "summary": "Brief explanation of classification",
    "key_findings": ["Aspect 1 covered", "Aspect 2 covered", "Aspect 3 covered"],
    "gaps_identified": ["Missing aspect 1 with details", "Missing aspect 2 with details"],
    "covered_aspects": ["Specific aspect 1 that IS covered", "Specific aspect 2 that IS covered"],
    "missing_aspects": ["Specific aspect 1 that is MISSING", "Specific aspect 2 that is MISSING"],
    "recommendations": ["Suggestion 1 to fill gap", "Suggestion 2 to fill gap"]
  },
  "metrics": {
    "total_candidates": number,
    "strong_supporters": number,
    "medium_supporters": number,
    "weak_supporters": number,
    "required_llrs": number
  }
}

IMPORTANT: For PARTIAL_TRACE classifications, you MUST populate:
- "gaps_identified": Detailed list of what's missing or any modifications compared to the HLR
- "missing_aspects": Specific HLR aspects not covered by any LLR or anything extra in the LLR which is not present in the HLR 
- "recommendations": Actionable suggestions for new LLRs needed
"""


def parse_reasoning_response(response: str) -> dict:
    """
    Parse reasoning response into structured format
    
    Args:
        response: Raw response from LLM
        
    Returns:
        Dictionary with boolean flags
    """
    result = {
        "is_required": False,
        "prevents_violation": False,
        "constrains_unsafe_behavior": False,
        "extends_beyond_intent": False,
        "raw_response": response
    }
    
    lines = response.strip().split('\n')
    for line in lines:
        line_upper = line.upper()
        if 'Q1:' in line or '1.' in line[:3]:
            result["is_required"] = 'YES' in line_upper
        elif 'Q2:' in line or '2.' in line[:3]:
            result["prevents_violation"] = 'YES' in line_upper
        elif 'Q3:' in line or '3.' in line[:3]:
            result["constrains_unsafe_behavior"] = 'YES' in line_upper
        elif 'Q4:' in line or '4.' in line[:3]:
            result["extends_beyond_intent"] = 'YES' in line_upper
    
    return result
