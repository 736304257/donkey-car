import time
WAIT_TIME = 0.03
class Sensor:
    """
    表示连接到Donkey Car的距离测量传感器类。
    """
    def __init__ (self, pi, range_gpios):
        """
        初始化超声波传感器驱动器
        """
        self.pi = pi
        if range_gpios is not None:
            from donkeycar.parts.hc_sr04 import Driver 
            self.range = Driver(self.pi, range_gpios[0], range_gpios[1])
        else:
            self.range = None
        self.distance = None

    def update_loop_body(self):
        """
        获取测量距离，并将其存储在实例变量距离中
        """
        self.distance = self.range.read()
        print('超声测距:',self.distance)
        time.sleep(WAIT_TIME)
        return self.distance

    def update (self):
        """
        实现在另一个线程中执行的处理。
       调用距离传感器类并定期将结果测量为实例变量distance并存储        
        """
        if self.range is not None:
            while True:
                self.update_loop_body()
        else:
            return None
    
    def run_threaded (self):
        #self.distance = self.range.read()
        return self.distance
    
    def run (self):
        if self.range is not None:
            self.distance = self.range.read()
        return self.distance
        
    def distance_get (self):
        if self.range is not None:
            self.distance = self.range.read()
            #print('超声测距:',self.distance)
            return self.distance
        elif self.range is None:
            return 11
    
    def shutdown (self):
        """
        关闭线程
        """
        if self.range is not None:
            self.range.cancel()
            self.range = None
            self.distance = None

