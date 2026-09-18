"""Make covenant-agent's sandbox able to run the generated theme worlds.

The corpus is built over ~104 themes that live as JSON under data/gen/themes/
and are registered at generation time, not baked into the world registry. Any
process that wants to execute one of those tasks has to register them first, or
every run dies on a missing world and it looks like a model failure.
"""
from __future__ import annotations

from corpus import COVENANT

_done = False


def register_themes() -> int:
    """Register every theme world. Safe to call more than once."""
    global _done
    if _done:
        from runtime.worlds import WORLDS
        return len(WORLDS)
    from data.gen.domains import register_domains
    from runtime.worlds import WORLDS
    register_domains(COVENANT / "data" / "gen" / "themes")
    _done = True
    return len(WORLDS)


if __name__ == "__main__":
    print(f"{register_themes()} worlds registered")
