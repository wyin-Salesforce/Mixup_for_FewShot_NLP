import statistics







# initializing list
test_list = [86.22, 88, 87.11, 83.77, 82.88]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
