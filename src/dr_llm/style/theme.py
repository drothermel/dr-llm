from __future__ import annotations


def width_to_tailwind(width: str) -> str:
    width_map = {
        "18rem": "w-72",
        "20rem": "w-80",
    }
    return width_map.get(width, width if width.startswith("w-") else "w-80")


__all__ = ["width_to_tailwind"]
