import statistics





# initializing list
test_list = [96.88, 96.88, 94.66, 96,97.11]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
