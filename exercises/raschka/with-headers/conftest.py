"""Make `import llms_from_scratch` work when running pytest without installing.
(You can still `pip install -e .` instead.)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pkg"))
