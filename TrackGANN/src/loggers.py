from functools import wraps
from collections import defaultdict
import os


def log_to_file(filename=None, header=None):
    ''' A decorator for logging function results, which accepts the directory path where 
    the log file will be saved and the filename as arguments. '''

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            log_filename = kwargs.pop('log_filename', filename)
            log_header = kwargs.pop('log_header', header)
            
            if log_filename and (not os.path.exists(log_filename) or os.path.getsize(log_filename) == 0):
                with open(log_filename, 'w') as f:
                    if log_header:
                        f.write(log_header + '\n')
            
            result = func(*args, **kwargs)
        
            if log_filename:
                with open(log_filename, 'a') as f:
                    if isinstance(result, tuple): 
                        f.write(','.join(map(str, result)) + '\n')
                    elif isinstance(result, defaultdict):
                        f.write(str(result) + '\n')
                    else: 
                        f.write(str(result) + ',')

            return result
        return wrapper
    return decorator

@log_to_file()
def get_time(start_time, end_time, **kwards):
    return end_time - start_time