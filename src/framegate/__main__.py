"""`python -m framegate > my_config.yaml` -- print a config template generated from the
GateConfig defaults (the single source of truth)."""

from .config import GateConfig

if __name__ == "__main__":
    print(GateConfig.to_yaml(), end="")
