import statistics





# initializing list
test_list = [
82.26,
81.42,
83.12,
79.99,
83.09
]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
