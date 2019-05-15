"""
shuffle csv file

"""
import pandas as pd


data = pd.read_csv('../datafile.csv', encoding='utf-8', engine='python')
#data = pd.read_csv('../tttttfile_2.csv', encoding='utf-8', engine='python')  # test용 file


print("***"*10)
print(len(data))


print("***"*10)
print(data)
print("***"*10)
sda = data.sample(frac=1)  # frec = traction, 얼마나 반활할껀지... 1= 전체 다 반환
#sdaa = dataff.sample(frac=1, replace=True)
print(sda)
#print(sdaa)


sda.to_csv('../shuffling_datafile.csv')
print("successful shuffling..")



