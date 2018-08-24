from datetime import datetime, timedelta


def timedelta_string(timedelta: timedelta) -> str:
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{}:{}:{}".format(hours, minutes, seconds)


def estimate_remaining_time(i, time, totalitems):
    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)
    return timedelta_string(estTime)
