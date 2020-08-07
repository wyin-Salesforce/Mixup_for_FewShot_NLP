import statistics







# initializing list
test_list = [97.33, 98.44, 96.88, 98, 92.66]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
