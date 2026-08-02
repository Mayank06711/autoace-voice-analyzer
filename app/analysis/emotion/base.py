"""Re-export the emotion interface/result so providers import from one place."""
from app.domain.contracts import EmotionProvider, EmotionResult

__all__ = ["EmotionProvider", "EmotionResult"]
