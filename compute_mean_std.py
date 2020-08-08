import statistics





# initializing list
test_list = [92, 93.11, 91.55, 91.55, 88.66]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
