# coach/types.py
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FeedbackReport:
    total_score: float
    components: Dict[str, float]
    issues: Dict[str, Any]
    recommendations: List[str]
    exercises: List[Dict[str, str]]
    segment_feedback: List[Dict[str, Any]]
    summary: Dict[str, Any]
