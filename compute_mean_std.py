import statistics




# initializing list
test_list = [83.11, 77.11, 88.44, 86.66, 79.11]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
