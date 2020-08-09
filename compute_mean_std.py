import statistics





# initializing list
test_list = [91.33, 94, 89.11, 88.22, 86]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
