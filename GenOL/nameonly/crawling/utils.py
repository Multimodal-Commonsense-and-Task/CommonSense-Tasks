import time
from datetime import datetime

def date_to_unix(date_obj: datetime.date) -> int:
    datetime_obj = datetime(date_obj.year, date_obj.month, date_obj.day)
    return int(time.mktime(datetime_obj.timetuple()))