import statistics





# initializing list
test_list = [88.22, 87.33, 88.66, 90, 89.55]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
