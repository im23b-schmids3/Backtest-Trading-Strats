from __future__ import annotations

import json
import logging
from pathlib import Path


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname, "event": record.getMessage()}, ensure_ascii=True)


def configure_logging(path: str | Path, level: str = "INFO") -> None:
    logger = logging.getLogger("research_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    target = str(Path(path))
    if any(isinstance(handler, logging.FileHandler) and handler.baseFilename == str(Path(target).resolve()) for handler in logger.handlers):
        return
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)

