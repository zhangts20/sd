import logging

__all__ = ["logger"]


class Logger:

    def __init__(self, name="Logger", log_level=logging.DEBUG):
        # Create a logger.
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Set the style of output.
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_loggger(self):
        return self.logger

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


logger = Logger()
