import statistics



# initializing list
test_list = [93.55, 94.44, 94.22, 93.77, 93.55]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
