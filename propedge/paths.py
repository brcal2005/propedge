"""Where propedge keeps its data. Override with PROPEDGE_HOME (e.g. a mounted disk when hosted)."""
import os


def home() -> str:
    return os.environ.get("PROPEDGE_HOME") or os.path.join(os.path.expanduser("~"), ".propedge")
