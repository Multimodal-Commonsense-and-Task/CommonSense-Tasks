import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from rich.logging import RichHandler

SILENT_LIBS = ["sentence_transformers", "httpx"]

LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT = "%m-%d %H:%M:%S"

NOTICE_LVL = 25
logging.addLevelName(NOTICE_LVL, "NOTICE")
def notice(self, message, *args, **kwargs):
    if self.isEnabledFor(NOTICE_LVL):
        self._log(NOTICE_LVL, message, args, **kwargs)
logging.Logger.notice = notice

IRM_LVL = 27
logging.addLevelName(IRM_LVL, "IRM")
def irm(self, message, *args, **kwargs):
    if self.isEnabledFor(IRM_LVL):
        self._log(IRM_LVL, message, args, **kwargs)
logging.Logger.irm = irm

def _add_handlers(log_path: str, level: int, max_bytes: int, backup: int, use_rich: bool):
    root = logging.getLogger()
    root.handlers.clear()

    if use_rich:
        sh = RichHandler(rich_tracebacks=True, markup=True, show_time=False)
    else:
        sh = logging.StreamHandler()
    
    sh.setFormatter(logging.Formatter(LOG_FMT, DATE_FMT))
    root.addHandler(sh)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(log_path, mode="a", maxBytes=max_bytes, backupCount=backup, encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FMT, DATE_FMT))
    root.addHandler(fh)
    root.setLevel(level)

def setup(log_path: str, level: int = logging.INFO, max_bytes: int = 1048576, backup: int = 5, use_rich: bool = True):
    _add_handlers(log_path, level, max_bytes, backup, use_rich)

    for lib in SILENT_LIBS:
        logging.getLogger(lib).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)