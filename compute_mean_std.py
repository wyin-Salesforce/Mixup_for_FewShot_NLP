import statistics





# initializing list
test_list = [83.52, 81.29, 80.39, 79.19, 80.16]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
