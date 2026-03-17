"""Model entrypoint for A10."""

from model_cls import BERTForClassification
from model_ner import BERTForNER

__all__ = ["BERTForClassification", "BERTForNER"]
