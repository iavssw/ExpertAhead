"""
Unified LLM W4A16 Quantized backend supporting LLaMA3.1-8B-Instruct.
"""

from .llama3_8b_w4a16_model import LLaMA3W4A16Model
from .mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model

__all__ = ["LLaMA3W4A16Model", "Mixtral8x7BW4A16Model"]

