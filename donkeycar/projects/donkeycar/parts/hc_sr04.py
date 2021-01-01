#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

import pigpio

LONGEST_TIME = 20000

class Driver:
    """
    该类封装了一种声学距离测量。
    """

    def __init__(self, pi, trigger, echo):
        """
        接收Pi对象使用，触发GPIO号和回显GPIO号。
        参数
             pi        gpio.pi对象
             trigger   触发Trig引脚所连接的GPIO编号
             echo      连接回波Echo引脚的GPIO编号
        返回值
            无
        """
        # 参数初始化
        self.pi    = pi
        self._trig = trigger
        self._echo = echo
        #是否使用echo来测量时间
        self._ping = False
        # echo引脚变为高电平的时间（微秒）
        self._high = None
        # echo引脚高电平持续的时间
        self._time = None

        # 是否被触发
        self._triggered = False

        # 在开始之前保存每个GPIO的状态
        self._trig_mode = pi.get_mode(self._trig)
        self._echo_mode = pi.get_mode(self._echo)

        # GPIO初始化
        pi.set_mode(self._trig, pigpio.OUTPUT)
        pi.set_mode(self._echo, pigpio.INPUT)

        # 设置Trig引脚的回调函数
        self._cb = pi.callback(self._trig, pigpio.EITHER_EDGE, self._cbf)
        # 设置Echo引脚的回调函数
        self._cb = pi.callback(self._echo, pigpio.EITHER_EDGE, self._cbf)

        # 是否被初始化
        self._inited = True

    def _cbf(self, gpio, level, tick):
        """
        回调函数
        引数
            gpio   0〜31   状态已更改的GPIO编号
            level   0〜2    0 =变为低（下降沿）
                            1 =变为高（上升沿）
                            2 =没有电平变化（看门狗超时）
            tick    32bit   启动后的微秒数
                            
        """
        # Trig引脚
        if gpio == self._trig:
            # 如果触发引脚是高电平，则触发发送
            if level == 0:
                # 处于已发送状态
                self._triggered = True
                # 消除Echo引脚变为高电平时的当前时间
                self._high = None
        # Echo引脚
        else:
            # 触发引脚发送了触发信息
            if self._triggered:
                # 如果echo引脚是高电平
                if level == 1:
                    # 存储当前时间（微妙）
                    self._high = tick
                # Echo引脚无变化或变低电平
                else:
                    
                    if self._high is not None:
                        # 计算Echo引脚的高电平时间（以微秒为单位）
                        self._time = tick - self._high
                        # 清除
                        self._high = None
                        # 使用Echo引脚进行定时
                        self._ping = True

    def read(self):
        """
        读数
         返回的读数将是声纳往返时间（微秒）。
         往返cms =往返时间 / 1000000.0 * 34030
        参数
            无
         返回值
             距离厘米距离
                             none表示超时或无法测量
        """
        # 如果已初始化
        if self._inited:
            # 不用echo来测量时间
            self._ping = False
            # 将触发信号发送到Trig引脚
            self.pi.gpio_trigger(self._trig, 11, pigpio.HIGH)
            # 存储触发信号发送到Trig引脚的时间
            start = time.time()
            # 循环直到Echo引脚的时间测量完成
            while not self._ping:
                #等待5秒钟或更长时间
                if (time.time()-start) > 5.0:
                    return duration_to_distance(LONGEST_TIME)
                # 休息0.001s
                time.sleep(0.001)
            # 返回距离
            return duration_to_distance(self._time)
        else:
            return None

    def cancel(self):
        """
        结束距离测量线程并将gpios返回原始模式。
        """
        if self._inited:
            self._inited = False
            self._cb.cancel()
            self.pi.set_mode(self._trig, self._trig_mode)
            self.pi.set_mode(self._echo, self._echo_mode)

def duration_to_distance(duration):
    """
    将Echo引脚为高电平的时间转换为距离。
    参数
       duration  回波引脚高电平的持续时间（微秒）
     返回值
         距离（cm）
                     none表示超时或无法测量
    """
    if duration is None or duration < 0 or duration == LONGEST_TIME:
        return None
    else:
        return (duration / 2.0) * 340.0 * 100.0 / 1000000.0

if __name__ == "__main__":

    RANGE_TRIG_GPIO = 5
    RANGE_ECHO_GPIO = 6
    RANGE_GPIOS = [
        RANGE_TRIG_GPIO,
        RANGE_ECHO_GPIO
    ]
    pi = pigpio.pi()
    sonar = Driver(pi, RANGE_GPIOS[0], RANGE_GPIOS[1])

    end = time.time() + 600.0

    r = 1
    while time.time() < end:

        print("{} {}".format(r, sonar.read()))
        r += 1
        time.sleep(0.03)

    sonar.cancel()

    pi.stop()
