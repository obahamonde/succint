import json
from typing import Any

from bs4 import BeautifulSoup, CData, Comment, Doctype, NavigableString, ResultSet, Tag


class SoupEncoder(json.JSONEncoder):
    """
    JSON encoder for BeautifulSoup objects.

    Overrides the default method of JSONEncoder to handle serialization of BeautifulSoup objects.
    It converts BeautifulSoup objects, including tags, navigable strings, comments, CDATA sections,
    and doctype declarations, into JSON-compatible representations.

    Args:
        o (Any): The object to be serialized.

    Returns:
        Any: The serialized representation of the object.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, (BeautifulSoup)):
            return [self.default(item) for item in o.find_all()]
        if isinstance(o, (CData, Comment, Doctype, NavigableString)):
            return {o.__class__.__name__.lower(): str(o)}
        if isinstance(o, (Tag)):
            return {
                o.__class__.__name__.lower(): {
                    "attrs": dict(o.attrs),
                    "children": [self.default(item) for item in o.find_all()],
                    "text": o.text,
                }
            }
        if isinstance(o, (ResultSet)):
            return [self.default(item) for item in o]  # type: ignore
        return super().default(o)
