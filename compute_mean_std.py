import statistics





# initializing list
test_list = [

72.12,
64.02,
73.15,
75.29

]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
