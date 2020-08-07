import statistics








# initializing list
test_list = [87.11, 89.11, 90.88, 86.88, 87.55]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
