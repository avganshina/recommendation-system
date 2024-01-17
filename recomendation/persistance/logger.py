import datetime

class Logger:
    def __init__(self):
        pass  # You can initialize any settings or configurations here

    def log(self, message):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {message}")