import statistics



# initializing list
test_list = [88.88, 91.11, 90.22, 83.33, 84.88]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
