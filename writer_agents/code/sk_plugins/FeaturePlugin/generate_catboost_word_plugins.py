#!/usr/bin/env python3
"""
Generate individual plugin files for each of the top 20 CatBoost success signal words.

This script creates individual plugin files following the existing plugin pattern.
"""

from pathlib import Path

# Top 20 words with their categories and success rates
TOP_20_WORDS = [
    ("order", "Motion Language", "50% success"),
    ("harm", "Endangerment Implicit", "Good"),
    ("safety", "Endangerment Implicit", "Good"),
    ("immediate", "Endangerment Implicit", "Good"),
    ("pseudonym", "Motion Language", "50% success"),
    ("risk", "Endangerment Implicit", "Good"),
    ("security", "National Security", "45.3% success"),
    ("serious", "Endangerment Implicit", "Good"),
    ("sealed", "Motion Language", "50% success"),
    ("motion", "Motion Language", "50% success"),
    ("citizen", "US Citizen Endangerment", "83.3% success"),
    ("complete", "Thoroughness", "Word count #1 predictor"),
    ("bodily", "Endangerment Implicit", "Good"),
    ("threat", "Endangerment Implicit", "Good"),
    ("impound", "Motion Language", "50% success"),
    ("protective", "Motion Language", "50% success"),
    ("national", "National Security", "45.3% success"),
]

PLUGIN_TEMPLATE = '''#!/usr/bin/env python3
"""
{word_title} Word Monitor Plugin - CatBoost Success Signal

Monitors usage of the word "{word}" which is a CatBoost success signal.
Category: {category}
Success Rate: {success_rate}
"""

import logging
from pathlib import Path
from typing import Optional

from semantic_kernel import Kernel
from .catboost_word_monitor_plugin import CatBoostWordMonitorPlugin

logger = logging.getLogger(__name__)


class {class_name}(CatBoostWordMonitorPlugin):
    """
    Plugin for monitoring the word "{word}" in motion documents.

    This word is identified as a CatBoost success signal:
    - Category: {category}
    - Success Rate: {success_rate}

    The plugin tracks frequency, context, and usage patterns to help
    calibrate with other plugins (sentences, paragraphs, arguments, etc.).
    """

    def __init__(
        self,
        kernel: Kernel,
        chroma_store=None,
        rules_dir: Optional[Path] = None,
        memory_store=None,
        **kwargs
    ):
        """Initialize {word_title} word monitor plugin."""
        super().__init__(
            kernel=kernel,
            word="{word}",
            word_category="{category}",
            success_rate="{success_rate}",
            chroma_store=chroma_store,
            rules_dir=rules_dir,
            memory_store=memory_store,
            **kwargs
        )
        logger.info("{class_name} initialized for monitoring word: '{word}'")
'''

def generate_plugin_file(word: str, category: str, success_rate: str, output_dir: Path):
    """Generate a plugin file for a specific word."""
    # Convert word to class name (e.g., "order" -> "OrderWordMonitorPlugin")
    word_title = word.title().replace('_', '')
    class_name = f"{word_title}WordMonitorPlugin"

    # Convert word to filename (e.g., "order" -> "order_word_monitor_plugin.py")
    filename = f"{word.lower().replace(' ', '_')}_word_monitor_plugin.py"
    filepath = output_dir / filename

    # Generate plugin content
    content = PLUGIN_TEMPLATE.format(
        word=word,
        word_title=word_title,
        class_name=class_name,
        category=category,
        success_rate=success_rate
    )

    # Write file
    filepath.write_text(content, encoding='utf-8')
    print(f"✅ Generated: {filename}")

    return class_name, filename

def main():
    """Generate all plugin files."""
    # Get the FeaturePlugin directory
    script_dir = Path(__file__).parent
    output_dir = script_dir

    print("🔧 Generating CatBoost word monitor plugins...\n")

    generated_plugins = []

    for word, category, success_rate in TOP_20_WORDS:
        class_name, filename = generate_plugin_file(word, category, success_rate, output_dir)
        generated_plugins.append((class_name, filename, word, category, success_rate))

    print(f"\n✅ Generated {len(generated_plugins)} plugin files")

    # Generate __init__.py additions
    print("\n📝 Add these imports to FeaturePlugin/__init__.py:")
    print("-" * 80)
    for class_name, filename, word, category, success_rate in generated_plugins:
        module_name = filename.replace('.py', '')
        print(f"from .{module_name} import {class_name}")

    print("\n📝 Add these to __all__ list:")
    print("-" * 80)
    for class_name, _, _, _, _ in generated_plugins:
        print(f'    "{class_name}",')

if __name__ == "__main__":
    main()

