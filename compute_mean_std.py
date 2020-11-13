import statistics





# initializing list
test_list = [

67.82,
81.79,
70.62,
80.52


]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
