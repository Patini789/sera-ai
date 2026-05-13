import logging
import coloredlogs
import os

def setup_logging(level='INFO'):
    """Sets up the global logging configuration."""
    log_format = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    
    # Apply coloredlogs to the root logger
    # This handles the root logger configuration without needing basicConfig
    coloredlogs.install(
        level=level,
        fmt=log_format,
        level_styles={
            'debug': {'color': 'white', 'faint': True},
            'info': {'color': 'cyan'},
            'warning': {'color': 'yellow', 'bold': True},
            'error': {'color': 'red', 'bold': True},
            'critical': {'color': 'red', 'bold': True, 'background': 'white'}
        },
        field_styles={
            'asctime': {'color': 'green'},
            'name': {'color': 'blue'},
            'levelname': {'color': 'magenta', 'bold': True}
        }
    )

    # Suppress noisy external loggers
    noisy_loggers = [
        'discord',
        'discord.client',
        'discord.gateway',
        'discord.http',
        'asyncio',
        'websockets',
        'werkzeug',
        'urllib3'
    ]
    
    # If the main level is INFO or higher, keep these at WARNING
    # If the main level is DEBUG, we can keep them at INFO to avoid absolute silence
    external_level = logging.WARNING if level != 'DEBUG' else logging.INFO
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(external_level)

def get_logger(name):
    """Returns a logger instance for the given name."""
    return logging.getLogger(name)
