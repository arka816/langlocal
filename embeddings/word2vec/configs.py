"""Configuration loader for Word2Vec training paths and settings."""

import os
from typing import Any, Dict
import yaml

config_dir: str = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(config_dir, "config.yaml"), "r") as f:
    configs: Dict[str, Any] = yaml.safe_load(f)

for key, value in configs['cache'].items():
    configs['cache'][key] = os.path.join(config_dir, value)
