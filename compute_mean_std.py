import statistics





# initializing list
test_list = [



74.59,
77.92,
78.32,
79.79




]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
