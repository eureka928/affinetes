"""Game rules loading module

This module provides functionality to load game rule descriptions for LLM prompts.
Game rules are stored in individual .txt files and cached for performance.
"""
import os
from pathlib import Path
from typing import Dict

_RULES_DIR = Path(__file__).parent
_RULES_CACHE: Dict[str, str] = {}


def get_game_rules(game_name: str) -> str:
    """
    Get game rules description for a specific game.
    
    Args:
        game_name: OpenSpiel game's short_name (e.g., "chess", "go", "leduc_poker")
        
    Returns:
        Rule description string. Returns empty string if no rule file exists.
        
    Note:
        Results are cached for performance. The cache persists for the lifetime
        of the Python process.
    """
    if game_name in _RULES_CACHE:
        return _RULES_CACHE[game_name]
    
    rules_file = _RULES_DIR / f"{game_name}.txt"
    if rules_file.exists():
        rules = rules_file.read_text(encoding='utf-8').strip()
        _RULES_CACHE[game_name] = rules
        return rules
    
    # No rules file found - return empty string
    return ""


def list_supported_games():
    """
    List all games that have rule files.
    
    Returns:
        List of game names (without .txt extension)
    """
    return sorted([f.stem for f in _RULES_DIR.glob("*.txt")])


def clear_cache():
    """Clear the rules cache. Useful for testing or reloading rules."""
    _RULES_CACHE.clear()