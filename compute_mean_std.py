import statistics




# initializing list
test_list = [79.87, 72.30, 79.87, 71.99, 71.96]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
