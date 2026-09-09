# apps/backend/src/models package
#
# Import every ORM model here so that Base.metadata is fully populated
# whenever this package is imported — required by Alembic env.py and by
# tests that call Base.metadata.create_all(engine).

from src.models.alert_settings import AlertSettings  # noqa: F401
from src.models.drift_report import DriftReport  # noqa: F401
from src.models.prediction import Prediction  # noqa: F401
from src.models.telegram_alert_lock import TelegramAlertLock  # noqa: F401
