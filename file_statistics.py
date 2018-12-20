import csv

labels = [0, 1, 2, 3, 4]
labels_count = [0, 0, 0, 0, 0]
count = 0
data = csv.reader(open('train.csv'))
next(data)  # Skip header row

for line in data: 
    count += 1
    for i in range(len(labels)):
        labels_count[int(line[3])] += 1
    total = sum(labels_count)
    for i in labels:
        print('Number of instances with ' + str(i) + ' label: ' + str(labels_count[i]))
    

