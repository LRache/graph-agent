"""Shared helpers for validating serialized core data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _required_mapping(
    data: Any,
    key: str,
    path: str,
) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} parent must be a JSON object")
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a JSON object")
    return value


def _required_list(data: Any, key: str, path: str) -> list[Any]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} parent must be a JSON object")
    value = data.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    return value


def _required_str(data: Any, key: str, path: str) -> str:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} parent must be a JSON object")
    value = data.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string")
    return value


def _required_int(data: Any, key: str, path: str) -> int:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} parent must be a JSON object")
    value = data.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{path} must be an integer")
    return value


def _optional_str(data: Any, key: str, path: str) -> str | None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} parent must be a JSON object")
    if key not in data or data[key] is None:
        return None
    return _required_str(data, key, path)


def _optional_bool(data: Any, key: str, path: str) -> bool | None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} parent must be a JSON object")
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    return value


def _optional_str_list(data: Any, key: str, path: str) -> list[str]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} parent must be a JSON object")
    value = data.get(key, [])
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{path} items must be strings")
        values.append(item)
    return values
