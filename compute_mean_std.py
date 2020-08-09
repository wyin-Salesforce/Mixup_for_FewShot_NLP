import statistics




# initializing list
test_list = [86.66, 88.44, 82.22, 92.88, 86.44]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
