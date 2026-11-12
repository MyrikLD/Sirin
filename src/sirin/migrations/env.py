from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, engine_from_config, pool

from sirin.tables import Base, UtcDateTime


def _render_item(type_: str, obj: object, autogen_context) -> str | bool:
    # Migrations stay independent of the application's types: UtcDateTime is stored as a plain DATETIME.
    if type_ == "type" and isinstance(obj, UtcDateTime):
        return "sa.DateTime()"
    return False


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        # SQLite alters a table by copying it into a new one.
        render_as_batch=True,
        render_item=_render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


# The application passes its open connection (Database.migrate); the alembic command line connects by itself.
if (connection := context.config.attributes.get("connection")) is not None:
    _run(connection)
else:
    if context.config.config_file_name is not None:
        fileConfig(context.config.config_file_name)
    engine = engine_from_config(
        context.config.get_section(context.config.config_ini_section, {}), poolclass=pool.NullPool
    )
    with engine.connect() as connection:
        _run(connection)
