import logging
import json
from logging.handlers import TimedRotatingFileHandler
import asyncio
import os
import traceback
import time

class Logger:
    def __init__(self, service_name, class_name, log_path, level=logging.INFO, buffer_size=100, flush_interval=1):
        self.service_name = service_name
        self.class_name = class_name
        self.log_path = log_path
        self.buffer_size = buffer_size
        self.logger = logging.getLogger(f"{service_name}.{class_name}")
        self.logger.setLevel(level)
        os.makedirs(self.log_path, exist_ok=True)
        log_file = os.path.join(self.log_path, f"{service_name}_{class_name}.log")
        self.handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=100)
        self.logger.addHandler(self.handler)
        self.flush_interval = flush_interval
        self.buffer = []
        self.flush_task = asyncio.ensure_future(self.flush_buffer())

    async def flush_buffer(self):
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush()

    async def flush(self):
        if self.buffer:
            log_records = '\n'.join(self.buffer)
            self.handler.emit(logging.LogRecord("", 0, "", 0, log_records, (), None))
            self.buffer.clear()

    def debug(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S,') + "{:03d}".format(int(time.time()*1000) % 1000)
        self.buffer.append(json.dumps({"timestamp": timestamp, "level": "DEBUG", "message": message}))
        if len(self.buffer) >= self.buffer_size:
            asyncio.ensure_future(self.flush())

    def info(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S,') + "{:03d}".format(int(time.time()*1000) % 1000)
        self.buffer.append(json.dumps({"timestamp": timestamp, "level": "INFO", "message": message}))
        if len(self.buffer) >= self.buffer_size:
            asyncio.ensure_future(self.flush())

    def warning(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S,') + "{:03d}".format(int(time.time()*1000) % 1000)
        self.buffer.append(json.dumps({"timestamp": timestamp, "level": "WARNING", "message": message}))
        if len(self.buffer) >= self.buffer_size:
            asyncio.ensure_future(self.flush())

    def error(self, message, exception):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S,') + "{:03d}".format(int(time.time()*1000) % 1000)
        exc_type = type(exception).__name__
        exc_msg = str(exception)
        tb_info = traceback.format_exc()
        
        # Accessing line number and file name directly from traceback object
        tb_lineno = exception.__traceback__.tb_lineno
        tb_filename = exception.__traceback__.tb_frame.f_code.co_filename
        
        # Split traceback_info by newline, strip whitespace, and remove empty entries
        tb_info_list = [line.strip() for line in tb_info.split('\n') if line.strip()]

        log_data = {
            "exception_type": exc_type,
            "exception_message": exc_msg,
            "traceback_info": tb_info_list,
            "line_number": tb_lineno,
            "file_name": tb_filename
        }
        self.buffer.append(json.dumps({"timestamp": timestamp, "level": "ERROR", "message": message, "details": log_data}))
        if len(self.buffer) >= self.buffer_size:
            asyncio.ensure_future(self.flush())

    def critical(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S,') + "{:03d}".format(int(time.time()*1000) % 1000)
        self.buffer.append(json.dumps({"timestamp": timestamp, "level": "CRITICAL", "message": message}))
        if len(self.buffer) >= self.buffer_size:
            asyncio.ensure_future(self.flush())

    def set_level(self, level):
        self.logger.setLevel(level)

    async def close(self):
        await self.flush()
        self.flush_task.cancel()
        self.handler.close()
