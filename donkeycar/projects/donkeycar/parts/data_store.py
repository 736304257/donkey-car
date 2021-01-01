import pickle
outputxmax = 0
outputxmin = 0
outputymax = 0
outputymin = 0
music_on = 0
with open("/home/pi/projects/donkeycar/parts/pos_data.txt", "wb") as f:
	pickle.dump(outputxmin, f)
	pickle.dump(outputymin, f)
	pickle.dump(outputxmax, f)
	pickle.dump(outputymax, f)
	pickle.dump(music_on, f)
	f.close()