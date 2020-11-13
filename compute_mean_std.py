import statistics





# initializing list
test_list = [

49.91,
50.48,
50.58,
50.31,
50.75,

]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
