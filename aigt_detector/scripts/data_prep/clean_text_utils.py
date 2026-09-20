"""
Unified Text Cleaning Utilities for 4chan AI Detection

This module provides consistent text cleaning functions used across:
1. Dataset generation (clean_4chan_data.py)
2. AI text generation (generate_aigt_text_v2.py)
3. Detection/inference (test_osm_det.py)

CRITICAL: All scripts must use the same cleaning pipeline to avoid
train/test mismatch and spurious correlations.

"""

import re
from typing import Optional


class FourchanTextCleaner:
    """
    Unified text cleaner for 4chan posts.

    Ensures consistent preprocessing across training and inference.
    """

    def __init__(self):
        """Initialize regex patterns for efficient cleaning."""
        self.patterns = {
            # Post references: >>123456
            'post_reference': re.compile(r'>>\d+'),

            # URLs: http://... or https://...
            'url': re.compile(r'https?://\S+'),

            # HTML tags: <br>, <span>, etc.
            'html_tag': re.compile(r'<[^>]+>'),

            # Quote markers at line start: ">some quoted text" or ">>other text"
            # This is CRITICAL for avoiding spurious correlations
            # Changed to >+ to match one or more > symbols (handles >>, >>>, etc.)
            'quote_marker': re.compile(r'^>+', re.MULTILINE),

            # Multiple spaces
            'multiple_spaces': re.compile(r' +'),

            # Multiple newlines
            'newlines': re.compile(r'\n+')
        }

    def clean(self, text: str, remove_quote_markers: bool = True) -> str:
        """
        Apply all cleaning steps to 4chan text.

        Args:
            text: Raw text from 4chan
            remove_quote_markers: If True, remove leading > symbols
                                  (RECOMMENDED to avoid spurious features)

        Returns:
            Cleaned text ready for AI detection
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Remove post references (>>123456)
        text = self.patterns['post_reference'].sub('', text)

        # Step 2: Remove URLs
        text = self.patterns['url'].sub('', text)

        # Step 3: Remove HTML tags
        text = self.patterns['html_tag'].sub('', text)

        # Step 4: Remove quote markers (>) at line start
        # This prevents the model from learning spurious correlations
        # where AI-generated text has different quote frequency than human text
        # Apply repeatedly to handle cases like "> >text" or ">> >text"
        if remove_quote_markers:
            prev_text = None
            while prev_text != text:
                prev_text = text
                text = self.patterns['quote_marker'].sub('', text)
                text = text.lstrip()  # Remove leading whitespace after each pass

        # Step 5: Convert newlines to spaces
        text = self.patterns['newlines'].sub(' ', text)

        # Step 6: Collapse multiple spaces
        text = self.patterns['multiple_spaces'].sub(' ', text)

        # Step 7: Strip leading/trailing whitespace
        text = text.strip()

        return text

    def is_ascii_art(self, text: str, threshold: float = 0.7) -> bool:
        """
        Detect ASCII art by checking letter ratio.

        Args:
            text: Input text
            threshold: Minimum ratio of letters (default: 0.7)

        Returns:
            True if likely ASCII art, False otherwise
        """
        if not text:
            return True

        # Count alphabetic characters
        letter_count = sum(1 for c in text if c.isalpha())
        total_count = len(text.replace(' ', '').replace('\n', ''))

        if total_count == 0:
            return True

        letter_ratio = letter_count / total_count
        return letter_ratio < threshold


# Singleton instance for easy import
cleaner = FourchanTextCleaner()


def clean_4chan_text(text: str, remove_quote_markers: bool = True) -> str:
    """
    Convenience function for cleaning 4chan text.

    Args:
        text: Raw 4chan text
        remove_quote_markers: Remove leading > symbols (default: True)

    Returns:
        Cleaned text

    Examples:
        >>> clean_4chan_text(">quoted text\\n>>123456\\nregular text")
        'quoted text regular text'

        >>> clean_4chan_text(">quoted text", remove_quote_markers=False)
        '>quoted text'
    """
    return cleaner.clean(text, remove_quote_markers=remove_quote_markers)


def is_ascii_art(text: str, threshold: float = 0.7) -> bool:
    """
    Convenience function for ASCII art detection.

    Args:
        text: Input text
        threshold: Minimum letter ratio

    Returns:
        True if likely ASCII art
    """
    return cleaner.is_ascii_art(text, threshold)
