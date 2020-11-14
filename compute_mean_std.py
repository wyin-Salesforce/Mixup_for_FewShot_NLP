import statistics





# initializing list
test_list = [



75.15,
69.55,
76.35,
76.39



]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
