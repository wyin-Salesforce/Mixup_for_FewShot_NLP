import statistics





# initializing list
test_list = [
82.69,
83.12,
84.59,
82.69,
84.69
]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
