import pickle
high_score = 7895
ID = 2
JJ = 14
DD = 14
money_s = 999
username = "lys916037541"
with open("/home/pi/projects/donkeycar/parts/savegame.txt", "wb") as f:
	pickle.dump(high_score, f)
	pickle.dump(ID, f)
	pickle.dump(JJ, f)
	pickle.dump(DD, f)
	pickle.dump(money_s, f)
	# pickle.dump(money, f)
	pickle.dump(username, f)
with open("/home/pi/projects/donkeycar/parts/savegame.txt", "rb") as f:
	
#	high_score = pickle.load(f)
#	ID = pickle.load(f)
#	JJ = pickle.load(f)
#	DD = pickle.load(f)
#	money_s = pickle.load(f)
	username = pickle.load(f)
#print(high_score)
#print(ID)
#print(JJ)
#print(DD)
#print(money_s)
print(username)