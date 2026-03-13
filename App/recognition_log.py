from datetime import datetime

logs = []

def add_log(name):

    logs.append({
        "name": name,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # keep only last 50 logs
    if len(logs) > 50:
        logs.pop(0)


def get_logs():
    return logs