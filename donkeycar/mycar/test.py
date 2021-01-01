import simpleaudio as sa
import wave
import numpy as np

filename = 'music1.mp3' 
with wave.open(filename ,'rb') as f:
    params = f.getparams()  #返回所有的WAV文件的格式信息，它返回的是一个元组(tuple)
    #它的前四项是：声道数,   量化位数（byte单位）, 采样频率,  总采样点数
    nchannels, sampwidth, framerate, nframes = params[:4]
    nframes = int(nframes)
    strData = f.readframes(nframes)  #二进制数据
    waveData = np.fromstring(strData)  #转换成np数组

#播放则要依靠simpleaudio库，把上面的参数传进来
play_obj = sa.play_buffer(waveData, nchannels, sampwidth,framerate)
play_obj.wait_done()  #这个函数的实质就是time.sleep()，等待音频播放结束