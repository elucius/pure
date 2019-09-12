#!D:\Soft\Apache24\htdocs1\v1\test\venv1\Scripts\python.exe
# -*- coding: UTF-8 -*-

import datetime, time
def unix_time(strtime: str):
    # 将python的datetime转换为unix时间戳
    # dtime = datetime.datetime.now()
    dtime=datetime.datetime.strptime(strtime, '%Y-%m-%d %H:%M')
    un_time = time.mktime(dtime.timetuple())
    # print(un_time)  #1509636609.0

    # 将unix时间戳转换为python  的datetime
    unix_ts = 1509636585.0
    times = datetime.datetime.fromtimestamp(unix_ts)
    # print(times) #2017-11-02 23:29:45
    return (un_time)

