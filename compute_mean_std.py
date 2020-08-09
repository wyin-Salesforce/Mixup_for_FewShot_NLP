import statistics



# initializing list
test_list = [96.66, 96.22, 96.22, 98.66, 97.11]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
