"""Pose plugins. At startup every module in this folder (except base.py and
names starting with "_") is imported, and every Pose subclass defined in it
with a `name` is instantiated - so adding a pose means adding one file here,
nothing else. See base.Pose for what a pose can define, and base.Gesture for
the per-action classes a pose's detector events map to.
"""
import importlib
import inspect
import pkgutil

from .base import Frame, Gesture, Pose, gmean, template

__all__ = ["Frame", "Gesture", "Pose", "gmean", "template", "load_poses"]


def load_poses() -> list:
    """One instance of every pose in this folder, sorted by (order, name)."""
    found = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_") or info.name == "base":
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            # defined in this module (not just imported into it), and named
            if issubclass(cls, Pose) and cls.__module__ == module.__name__ and cls.name:
                found.append(cls())

    names = [p.name for p in found]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise RuntimeError(f"duplicate pose names in poses/: {dupes}")
    if "none" in names:
        raise RuntimeError('"none" is reserved for "no pose matched" - rename that pose')
    return sorted(found, key=lambda p: (p.order, p.name))
