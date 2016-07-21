#!usr/bin/env python
#coding:utf-8

import time
import datetime


class DateConvertor(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_timestamp(timestamp, convert_format="month"):
        if convert_format == "month" or convert_format == "m":
            return DateConvertor.convert_month_timestamp(timestamp)
        elif convert_format == "week" or convert_format == "w":
            return DateConvertor.convert_week_timestamp(timestamp)
        elif convert_format == "day" or convert_format == "d":
            return DateConvertor.convert_day_timestamp(timestamp)
        else:
            raise IOError

    @staticmethod
    def convert_string(timestamp):
        year_month_day = time.localtime(timestamp)
        year, month, day = year_month_day.tm_year, year_month_day.tm_mon, year_month_day.tm_mday
        hour, minute, second = year_month_day.tm_hour, year_month_day.tm_min, year_month_day.tm_sec
        return '%d-%d-%d %d:%d:%d' % (year, month, day, hour, minute, second)

    @staticmethod
    def convert_month_timestamp(timestamp):
        year_month_day = datetime.datetime.fromtimestamp(timestamp)
        month_timestamp = datetime.datetime(year=year_month_day.year, month=year_month_day.month, day=1)
        return time.mktime(month_timestamp.timetuple())

    @staticmethod
    def convert_week_timestamp(timestamp):
        week = int(time.strftime('%w', time.localtime(timestamp)))
        MondayStamp = timestamp - (week-1)*86400
        MondayStr = time.localtime(MondayStamp)
        return time.mktime(time.strptime(time.strftime('%Y-%m-%d', MondayStr), '%Y-%m-%d'))

    @staticmethod
    def convert_day_timestamp(timestamp):
        year_month_day = datetime.datetime.fromtimestamp(timestamp)
        month_timestamp = datetime.datetime(year=year_month_day.year, month=year_month_day.month, day=year_month_day.day)
        return time.mktime(month_timestamp.timetuple())

if __name__ == '__main__':
    now = time.mktime(time.localtime())
    print DateConvertor.convert_timestamp(now, 'm')
    print DateConvertor.convert_string(now)