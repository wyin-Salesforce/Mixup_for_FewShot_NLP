import statistics





# initializing list
test_list = [

87.06,
87.62,
87.79

]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
