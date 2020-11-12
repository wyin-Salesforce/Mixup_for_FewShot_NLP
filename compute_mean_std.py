import statistics





# initializing list
test_list = [
83.22,
81.09,
79.19,
82.79,
83.86
]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
