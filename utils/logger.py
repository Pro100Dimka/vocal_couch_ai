import logging


def build_logger(name="trainer", path="train.log"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # если уже есть handlers — не дублируем, но и не блокируем изменение файла
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


logger = build_logger()


def log(msg, var=None):
    msg = f"{msg} {var}"
    logger.info(msg)
