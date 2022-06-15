import time
import os

__all__ = ['Logger']


class Logger(object):
    def __init__(self, task_name, stream, path):
        # Add path check
        path = path.strip()
        if not os.path.exists(path):
            os.mkdir(path)
            
        self.file_name = path + task_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".log"
        self.terminal = stream
        self.log = open(self.file_name, 'a+')#TODO: comment out just momentaneously

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)#TODO: comment out just momentaneously

    def flush(self):
	    pass
