import logging

def get_logger(name: str) -> logging.Logger:
    # 创建logger实例
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.hasHandlers():
        # 创建file handler
        file_handler = logging.FileHandler(f'logs/{name}.log')
        file_handler.setLevel(logging.WARNING)

        # 创建console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 创建formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger