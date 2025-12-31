"""
Configuration for HLR Pipeline
Adjust these settings to tune rate limiting and API behavior
"""

# ============================================================================
# API Rate Limiting Configuration
# ============================================================================

# Groq API Rate Limits (adjust based on your plan)
GROQ_MIN_REQUEST_INTERVAL = 0.8  # Seconds between requests (slightly conservative)
GROQ_MAX_RETRIES = 3
GROQ_RETRY_DELAY = 10  # Seconds to wait on rate limit error

# No longer using Gemini - all via Groq
USE_GEMINI = False

# ============================================================================
# Embedding Configuration
# ============================================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation

# ============================================================================
# Classification Configuration
# ============================================================================

# Similarity thresholds (tuned for better discrimination)
SIMILARITY_THRESHOLD_VERY_STRONG = 0.85  # Very strong relationship (LLR01-03 → HLR01)
SIMILARITY_THRESHOLD_STRONG = 0.70       # Strong relationship
SIMILARITY_THRESHOLD_MEDIUM = 0.45       # Medium relationship (raised from 0.50)
SIMILARITY_THRESHOLD_WEAK = 0.30         # Weak but valid relationship (lowered from 0.35)

# Top-K candidate retrieval
TOP_K_CANDIDATES = 10  # Number of LLR candidates to retrieve per HLR

# Classification thresholds for HLRs (stricter requirements)
MIN_VERY_STRONG_FULLY_TRACED = 3       # Need 3 very strong LLRs for FULLY_TRACED
MIN_STRONG_TOTAL_PARTIAL_TRACE = 1     # 1-2 strong = PARTIAL_TRACE
MAX_STRONG_PARTIAL_TRACE = 2           # More than 2 strong → FULLY_TRACED

# Weight computation factors
WEIGHT_SIMILARITY_FACTOR = 0.6         # How much similarity contributes to weight
WEIGHT_REASONING_FACTOR = 0.4          # How much reasoning contributes to weight

# ============================================================================
# LLM Model Configuration
# ============================================================================

# Groq models for different tasks
GROQ_REASONING_MODEL = "openai/gpt-oss-120b"  # For reasoning about requirements
GROQ_STRUCTURED_MODEL = "llama-3.1-8b-instant"  # For JSON formatting/conclusion
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"  # Fallback if others fail

LLM_TEMPERATURE = 0.2  # Lower = more deterministic
LLM_MAX_TOKENS = 1000

# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_DIR = "../dist"
OUTPUT_FILE = "hlrs-summary.json"
GRAPH_FILE = "hlr-llr-graph.png"

# Visualization settings
GRAPH_FIGURE_SIZE = (16, 12)
GRAPH_DPI = 150
GRAPH_NODE_SIZE_HLR = 3000
GRAPH_NODE_SIZE_LLR = 2000

# ============================================================================
# Performance Configuration
# ============================================================================

# Enable parallel processing (experimental)
ENABLE_PARALLEL = False  # Set to True to process HLRs in parallel
MAX_WORKERS = 2  # Number of parallel workers (if ENABLE_PARALLEL=True)

# Progress bar settings
SHOW_PROGRESS_BARS = True
PROGRESS_BAR_REFRESH_RATE = 10  # Updates per second

# ============================================================================
# Debug Configuration
# ============================================================================

VERBOSE = False  # Enable verbose logging
SAVE_INTERMEDIATE_RESULTS = False  # Save reasoning outputs for debugging
DEBUG_MODE = False  # Enable debug mode with extra checks

# ============================================================================
# API Plan Presets
# ============================================================================

def set_groq_plan(plan: str):
    """
    Adjust Groq rate limits based on plan
    
    Options:
    - 'free': Free tier (14400 RPD, 30 RPM)
    - 'pay-as-you-go': Pay as you go (14400 RPD, 30 RPM)
    - 'developer': Developer plan (7200000 TPD, 600 RPM)
    """
    global GROQ_MIN_REQUEST_INTERVAL
    
    if plan == 'free' or plan == 'pay-as-you-go':
        GROQ_MIN_REQUEST_INTERVAL = 2.0  # 30 RPM = 1 request per 2 seconds
    elif plan == 'developer':
        GROQ_MIN_REQUEST_INTERVAL = 0.1  # 600 RPM = 1 request per 0.1 seconds
    else:
        print(f"Unknown plan: {plan}, using default")


def set_gemini_plan(plan: str):
    """
    Adjust Gemini rate limits based on plan
    
    Options:
    - 'free': Free tier (15 RPM, 1500 RPD)
    - 'pay-as-you-go': Pay as you go (360 RPM, 10000 RPD)
    """
    global GEMINI_MIN_REQUEST_INTERVAL
    
    if plan == 'free':
        GEMINI_MIN_REQUEST_INTERVAL = 4.0  # 15 RPM = 1 request per 4 seconds
    elif plan == 'pay-as-you-go':
        GEMINI_MIN_REQUEST_INTERVAL = 0.2  # 360 RPM = 1 request per 0.17 seconds
    else:
        print(f"Unknown plan: {plan}, using default")


# ============================================================================
# Load configuration from environment (optional)
# ============================================================================

def load_from_env():
    """Load configuration overrides from environment variables"""
    import os
    
    global GROQ_MIN_REQUEST_INTERVAL, GEMINI_MIN_REQUEST_INTERVAL
    global TOP_K_CANDIDATES, VERBOSE
    
    if os.getenv("GROQ_RATE_LIMIT"):
        GROQ_MIN_REQUEST_INTERVAL = float(os.getenv("GROQ_RATE_LIMIT"))
    
    if os.getenv("GEMINI_RATE_LIMIT"):
        GEMINI_MIN_REQUEST_INTERVAL = float(os.getenv("GEMINI_RATE_LIMIT"))
    
    if os.getenv("TOP_K"):
        TOP_K_CANDIDATES = int(os.getenv("TOP_K"))
    
    if os.getenv("VERBOSE"):
        VERBOSE = os.getenv("VERBOSE").lower() == "true"
