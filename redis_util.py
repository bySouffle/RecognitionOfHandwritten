"""
    redis计数模块
"""


import redis
from datetime import date


# 增加使用redis统计访问次数的功能



REDIS_HOST = "127.0.0.1"  # redis host
MNIST_KEY = "MNIST"  # 总访问次数
TODAY_KEY = "TODAY"  # 今日日期
TODAY_TIME_KEY = "TODAY_TIME"  # 今日访问次数

# 初始化redis
r = redis.StrictRedis(host=REDIS_HOST)

# 8390
# r.set(MINIST_KEY, 8390)
# r.set(TODAY_TIME_KEY, 1)
# 从redis中获取当前的访问次数
r.get(MNIST_KEY)

# 增加访问次数
def inc_visit_num():
    r.incr(MNIST_KEY)
    if (get_today().encode() == r.get(TODAY_KEY)):
        r.incr(TODAY_TIME_KEY)
    else:
        r.set(TODAY_KEY, get_today().encode())
        r.set(TODAY_TIME_KEY, 1)

# 获取总访问次数
def get_visit_num_all():
    return r.get(MNIST_KEY).decode()

# 获取今日访问次数
def get_visit_num_today():
    return r.get(TODAY_TIME_KEY).decode()

# 获取今日日期
def get_today():
    return date.today().strftime('%Y-%m-%d')



if __name__ == '__main__':
    inc_visit_num() # 访问一次
    print(get_visit_num_all())
    print(get_visit_num_today())
   
