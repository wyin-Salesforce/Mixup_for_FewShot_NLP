import statistics




# initializing list
test_list = [91.33, 92.88, 88.22, 90.88, 91.77]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
