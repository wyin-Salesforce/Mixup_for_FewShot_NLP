import statistics





# initializing list
test_list = [


4.60,
3.45,
2.85,
2.73



]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
