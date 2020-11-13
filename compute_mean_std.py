import statistics





# initializing list
test_list = [




82.22,
79.39,
80.76,
79.35,
78.65




]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
