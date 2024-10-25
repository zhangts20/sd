import logging
import colorlog

from datetime import datetime
from zoneinfo import ZoneInfo

__all__ = ["logger"]


class BeijingTimeFormatter(colorlog.ColoredFormatter):

    LEVEL_SHORTNAMES = {
        'DEBUG': 'D',
        'INFO': 'I',
        'WARNING': 'W',
        'ERROR': 'E',
        'CRITICAL': 'C'
    }

    def formatTime(self, record, datefmt=None):
        utc_dt = datetime.fromtimestamp(record.created, tz=ZoneInfo('UTC'))
        beijing_dt = utc_dt.astimezone(ZoneInfo('Asia/Shanghai'))
        if datefmt:
            return beijing_dt.strftime(datefmt)
        else:
            return beijing_dt.strftime("%m/%d/%Y-%H:%M:%S")

    def format(self, record):
        record.levelname = self.LEVEL_SHORTNAMES.get(record.levelname,
                                                     record.levelname)
        return super().format(record)


class Logger:

    def __init__(self, name="SD", log_level=logging.DEBUG):
        # create a logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # set the type of output
        handler = logging.StreamHandler()
        formatter = BeijingTimeFormatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s]%(log_color)s %(message)s",
            datefmt="%m/%d/%Y-%H:%M:%S",
            log_colors={
                "D": "cyan",
                "I": "green",
                "W": "yellow",
                "E": "red",
                "C": "bold_red",
            })
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def set_level(self, level):
        self.logger.setLevel(level)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


sd_logger = Logger()
