from logging import DEBUG, Formatter, Logger, StreamHandler, getLogger

logger: Logger = getLogger(__name__)
logger.setLevel(DEBUG)

# create console handler and set level to debug
ch = StreamHandler()
ch.setLevel(DEBUG)

# create formatter
formatter = Formatter("%(funcName)s (%(module)s) %(levelname)s %(message)s")

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


def log(*values, sep: str | None = " ") -> None:
    used_sep: str
    if sep is None:
        used_sep = " "
    else:
        used_sep = sep
    logger.info(used_sep.join(values), stacklevel=2)
