import statistics


# initializing list
test_list = [90.88, 95.33, 94.88, 93.55, 94]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
