# apps/backend/src/models package
#
# Import every ORM model here so that Base.metadata is fully populated
# whenever this package is imported — required by Alembic env.py and by
# tests that call Base.metadata.create_all(engine).

from src.models.prediction import Prediction  # noqa: F401
