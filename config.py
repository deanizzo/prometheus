# config.py
import json
from pathlib import Path
import math

class Section:
    """Simple object that allows attribute-style access."""
    def __init__(self, **entries):
        for key, value in entries.items():
            setattr(self, key, value)

class Config:
    def __init__(self, json_path: Path):
        with open(json_path) as f:
            data = json.load(f)

        # Convert each section into attribute-access objects
        self.field = Section(**data["field_variables"])
        self.sim = Section(**data["simulation_variables"])
        self.csv = Section(**data["csv_variables"])

        # Derived values become real attributes
        self.field.receiver_radius = self.field.receiver_diameter / 2
        self.field.receiver_circumference = (
            2 * math.pi * self.field.receiver_radius
        )

# Resolve path relative to this file
json_path = "config/config.json"

# Singleton instance
cfg = Config(json_path)
print(f"Configuration loaded from {json_path}")
