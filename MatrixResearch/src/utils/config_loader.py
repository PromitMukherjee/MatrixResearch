import yaml
import os

def load_config(config_path):
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_experiment_config(base_config_path="configs/default.yaml", exp_config_path=None):
    """Loads default config and merges experiment specific config."""
    config = load_config(base_config_path)
    if exp_config_path and os.path.exists(exp_config_path):
        exp_config = load_config(exp_config_path)
        # Simple top-level merge
        for k, v in exp_config.items():
            if k in config and isinstance(config[k], dict) and isinstance(v, dict):
                config[k].update(v)
            else:
                config[k] = v
    return config
