import scipy.io
import numpy as np

train_label = np.load('data/testdata.npy')

with open("testdata_res_row.txt", "w") as f:
    for i in range(0, 455000):
        a_list = train_label[i, :].tolist()
        # print(a)
        set(a_list)
        res_0 = a_list.count(0)
        res_1 = a_list.count(1)
        #print('0:', res_0, '1:', res_1)

        res_row = str(res_0) + '    ' + str(res_1) + '\n'
        f.write(res_row)
i = 0
#print(a_list)
a_list = []
print(a_list)

with open("testdata_res_column.txt", "w") as f:
    for i in range(0, 919):
        a_list = train_label[:, i].tolist()
        # print(a)
        set(a_list)
        res_0 = a_list.count(0)
        res_1 = a_list.count(1)
        #print('0:', res_0, '1:', res_1)

        res_column = str(res_0) + '    ' + str(res_1) + '\n'
        f.write(res_column)

'''
sum = 0
with open("sum_126_full.txt", "w") as f:
    for j in range(0, 4400000):
        if(train_label[j, 126] == 1):
            corr = str(j) + ','
            f.write(corr)

    #f.write('\n')
'''
'''
sum_list = []
less = 0
for i in range(0, 919):
    a_list = train_label[:, i].tolist()
    # print(a)
    set(a_list)
    res_1 = a_list.count(1)
    if res_1 < 10000:
        less +=1
        sum_list.append(i)
with open("sort_num_res_full.txt", "w") as f:
    f.write(str(sum_list))
print(less)

num_list = []
with open('data_126.txt','w') as f:
    for i in range(0, 4400000):
        if train_label[i, 126] == 1:
            num_list.append(i)
    f.write(str(num_list))

print('OK!')
'''
'''
result = []
with open('data_126.txt','r') as f:
    for line in f.readlines():
        tmp = line.split(', ')
        for i in range(0, len(tmp)):
            result.append(tmp[i])

result = list(map(int, result))
#print(result)
'''


