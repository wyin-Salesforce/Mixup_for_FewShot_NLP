import statistics






# initializing list
test_list = [70.66, 79.55, 77.11, 83.33, 79.33]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
