"""Neo4j connection resolution, decoupled from schema.

A `Neo4jConnection` is a (uri, user, password) triple plus a short label for
logging. It is **independent** of `SchemaSpec` — the same schema can run
against multiple Neo4j instances by passing different connections to the
adapter.

Default targets (named):

- ``default`` reads ``NEO4J_URI`` / ``NEO4J_USERNAME`` / ``NEO4J_PASSWORD``.
  This is the project's own Aura instance (the ``soros`` instance per
  `.mcp.json`).
- ``yhoga`` reads ``NEO4J_URI_YHOGA`` / ``NEO4J_USERNAME_YHOGA`` /
  ``NEO4J_PASSWORD_YHOGA``. The peer-owned Yhoga Aura instance.

Selecting the target:

1. If a `Neo4jConnection` is passed explicitly to `get_adapter(...)` or to
   `SchemaAdapter(connection=...)`, that wins.
2. Otherwise, if the ``NEO4J_TARGET`` env var is set to a known target name,
   that connection is used for every adapter regardless of schema.
3. Otherwise the adapter falls back to ``SchemaSpec``'s default env-var names
   (`spec.neo4j_uri_env` etc.) — the legacy "yhoga schema implies yhoga DB"
   behavior, preserved for backward compatibility.

The intended common case for cross-schema experimentation is (1) or (2): the
user picks yhoga schema in the sidebar but sets ``NEO4J_TARGET=default`` (or
selects "default" in the sidebar) so the yhoga ontology runs against their
own upstream.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Neo4jConnection:
    """Concrete Neo4j connection parameters, resolved from env or explicit."""

    uri: str
    user: str
    password: str
    label: str = "default"

    @classmethod
    def from_env(
        cls,
        uri_env: str = "NEO4J_URI",
        user_env: str = "NEO4J_USERNAME",
        pwd_env: str = "NEO4J_PASSWORD",
        label: str = "default",
    ) -> "Neo4jConnection":
        uri = os.getenv(uri_env)
        user = os.getenv(user_env, "neo4j")
        pwd = os.getenv(pwd_env)
        if not uri or not pwd:
            raise ValueError(
                f"{uri_env} and {pwd_env} must be set for connection '{label}'"
            )
        return cls(uri=uri, user=user, password=pwd, label=label)

    @classmethod
    def default(cls) -> "Neo4jConnection":
        """Project default — reads NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD."""
        return cls.from_env(label="default")

    @classmethod
    def yhoga(cls) -> "Neo4jConnection":
        """Yhoga upstream — reads NEO4J_URI_YHOGA / NEO4J_USERNAME_YHOGA / NEO4J_PASSWORD_YHOGA."""
        return cls.from_env(
            uri_env="NEO4J_URI_YHOGA",
            user_env="NEO4J_USERNAME_YHOGA",
            pwd_env="NEO4J_PASSWORD_YHOGA",
            label="yhoga",
        )

    @classmethod
    def by_name(cls, name: str) -> "Neo4jConnection":
        """Resolve a named target. Currently 'default' or 'yhoga'.

        Aliases accepted: 'soros' → 'default'.
        """
        n = (name or "").lower()
        if n in ("default", "soros", ""):
            return cls.default()
        if n == "yhoga":
            return cls.yhoga()
        raise ValueError(
            f"Unknown Neo4j target '{name}'. Known: {sorted(CONNECTION_TARGETS)}"
        )


# Display-name → factory mapping; used by sidebar selectors.
CONNECTION_TARGETS: dict[str, str] = {
    "default": "Default Neo4j (NEO4J_URI — your own upstream)",
    "yhoga":   "Yhoga upstream (NEO4J_URI_YHOGA)",
}


def resolve_connection(explicit: Neo4jConnection | None = None) -> Neo4jConnection | None:
    """Pick the active connection or return None to use spec defaults.

    Resolution order:
      1. `explicit` argument (highest priority).
      2. `NEO4J_TARGET` env var if set to a known target name.
      3. None — caller should fall back to `SchemaSpec`'s env-var names.
    """
    if explicit is not None:
        return explicit
    target = os.getenv("NEO4J_TARGET")
    if target:
        return Neo4jConnection.by_name(target)
    return None
