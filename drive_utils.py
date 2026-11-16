"""Helpers for accepting Google Drive model references from the UI."""

import re
from urllib.parse import parse_qs, urlparse


_FILE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{10,}")
_ALLOWED_HOSTS = {"drive.google.com", "docs.google.com"}


def extract_google_drive_file_id(value):
    """Return a Drive file ID from a raw ID or supported Google Drive URL."""
    candidate = value.strip()
    if _FILE_ID_PATTERN.fullmatch(candidate):
        return candidate

    parsed = urlparse(candidate)
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError("Enter a Google Drive share link or file ID.")

    parts = [part for part in parsed.path.split("/") if part]
    if "d" in parts:
        index = parts.index("d")
        if index + 1 < len(parts) and _FILE_ID_PATTERN.fullmatch(parts[index + 1]):
            return parts[index + 1]

    query_id = parse_qs(parsed.query).get("id", [None])[0]
    if query_id and _FILE_ID_PATTERN.fullmatch(query_id):
        return query_id

    raise ValueError("The Google Drive link does not contain a valid file ID.")
