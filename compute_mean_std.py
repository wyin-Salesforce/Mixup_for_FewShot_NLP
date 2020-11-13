import statistics





# initializing list
test_list = [

50.31,
52.65,
52.98,
49.68

]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
