import statistics





# initializing list
test_list = [

85.21,
84.03,
85.24,
85.59



]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
