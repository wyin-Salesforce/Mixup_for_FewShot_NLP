import statistics





# initializing list
test_list = [
85.06,
79.72,
83.76,
83.86,
83.09
]
print('sum:', sum(test_list))
average = round(sum(test_list)/len(test_list), 2)
res = round(statistics.pstdev(test_list),2)

print(str(average)+'/'+str(res))
