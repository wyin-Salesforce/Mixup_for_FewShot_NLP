import statistics



# initializing list
test_list = [87.77, 91.55, 86.44, 89.77, 80.44]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
