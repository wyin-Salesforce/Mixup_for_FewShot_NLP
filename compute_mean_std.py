import statistics



# initializing list
test_list = [92.22, 91.77, 89.33, 89.77, 86.88]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
