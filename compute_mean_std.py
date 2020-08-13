import statistics





# initializing list
test_list = [50.51, 51.51, 50.15, 50.95, 50.31]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
