import statistics





# initializing list
test_list = [82.26, 84.69, 79.69, 83.76, 82.42]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
