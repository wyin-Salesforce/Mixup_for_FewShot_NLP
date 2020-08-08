import statistics




# initializing list
test_list = [91.11, 87.55, 86.44, 84.22, 82]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
