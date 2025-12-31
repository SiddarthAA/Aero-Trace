"""
LLR Classification Package
Analyzes LLRs in isolation to determine their justification
"""

from .llr_classifier import LLRClassifier
from .prompts import get_llr_reasoning_prompt, get_llr_classification_prompt

__all__ = ['LLRClassifier', 'get_llr_reasoning_prompt', 'get_llr_classification_prompt']
