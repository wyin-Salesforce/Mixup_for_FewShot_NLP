import statistics




# initializing list
test_list = [87.55,84.22,86.66,84.88,80.88
]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
