import statistics


# initializing list
test_list = [82.44, 85.11, 85.77, 92.44, 90.88]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
