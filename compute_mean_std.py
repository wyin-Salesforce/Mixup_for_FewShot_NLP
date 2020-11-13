import statistics





# initializing list
test_list = [


82.36,
80.89,
83.29,
81.46






]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
