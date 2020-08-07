import statistics





# initializing list
test_list = [87.11, 82.88, 86.66, 91.55, 88.44]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
