"""Normalize database URLs for SQLAlchemy with psycopg."""

from sqlalchemy.engine import make_url


def sqlalchemy_dsn(dsn: str) -> str:
    """Return a DSN string for SQLAlchemy, appending ``+psycopg`` when no driver is set."""
    url = make_url(dsn)
    if "+" in url.drivername:
        return url.render_as_string(hide_password=False)
    base = url.drivername
    if base == "postgres":
        base = "postgresql"
    return url.set(drivername=f"{base}+psycopg").render_as_string(hide_password=False)
