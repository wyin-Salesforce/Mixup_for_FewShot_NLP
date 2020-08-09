import statistics




# initializing list
test_list = [98.44, 98.22, 98.44, 98.22, 97.55]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
