"""
Helper utilities for loading Semantic Kernel prompt-based functions.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional

from semantic_kernel import Kernel


def load_semantic_function(
    kernel: Kernel,
    package_path: str,
    function_directory: str,
    prompt_file: str = "skprompt.txt",
    config_file: str = "config.json",
):
    """
    Load an SK semantic function from prompt/config files bundled in a package.
    """

    with resources.as_file(resources.files(package_path) / function_directory) as fn_dir:
        prompt_path = Path(fn_dir) / prompt_file
        config_path = Path(fn_dir) / config_file
        if not prompt_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Semantic function files missing in {fn_dir}: expected "
                f"{prompt_file} and {config_file}."
            )
        with prompt_path.open("r", encoding="utf-8") as prompt_file_handle:
            prompt_template = prompt_file_handle.read()
        with config_path.open("r", encoding="utf-8") as config_file_handle:
            config_text = config_file_handle.read()

    return kernel.create_semantic_function(
        prompt_template,
        plugin_name=None,
        function_name=None,
        prompt_config=config_text,
    )

