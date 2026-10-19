"""Simple configuration loader for video-robot-agent."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, tries:
                    1. config.yaml
                    2. configs/config.yaml
                    3. configs/default_config.yaml

    Returns:
        Dictionary with configuration
    """
    if config_path and Path(config_path).exists():
        path = Path(config_path)
    else:
        # Try to find config file
        for p in ['config.yaml', 'configs/config.yaml', 'configs/default_config.yaml']:
            if Path(p).exists():
                path = Path(p)
                break
        else:
            raise FileNotFoundError("No config file found. Create config.yaml or use configs/default_config.yaml")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # Replace environment variables
    def replace_env_vars(obj):
        if isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            var_name = obj[2:-1]
            return os.environ.get(var_name, obj)
        elif isinstance(obj, dict):
            return {k: replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_env_vars(item) for item in obj]
        return obj

    config = replace_env_vars(config)
    return config

# Load config once when module is imported
BINDER_CONFIG = load_config()