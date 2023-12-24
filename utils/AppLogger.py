import logging

class AppLogger:
    def __init__(self, moduleName, logfile):
        self._logger = logging.getLogger(moduleName)
        handler = logging.FileHandler(logfile)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self._logger.addHandler(console)

        self.error = self._logger.error
        self.info = self._logger.info
        self.debug = self._logger.debug