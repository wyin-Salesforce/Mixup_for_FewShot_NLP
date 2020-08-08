import statistics



# initializing list
test_list = [81.33, 85.33, 85.11, 82.88, 73.55]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
