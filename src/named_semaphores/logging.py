import logging


class LoggingMixin:
    def __init__(self, name: str, level: int = logging.INFO) -> None:
        """
        Initialize the LoggingMixin class.

        :param str name: The name of the logger.
        :param int level: The logging level. Default is logging.INFO.
        """
        name = name if isinstance(name, str) else ""
        self.logger = self.create_logger(name, level)

    def create_logger(self, name: str, level: int = logging.INFO) -> logging.Logger:
        """
        Create a logger with the given name and level. Class name is prepended to the logger name.

        :param str name: The name of the logger.
        :param int level: The logging level. Default is logging.INFO.
        :return: The created logger.
        :rtype: logging.Logger
        """
        logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False  # Avoid propagating to the root logger
        return logger
