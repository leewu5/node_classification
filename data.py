import numpy as np
from collections import namedtuple

Data = namedtuple('Data', ['x', 'y', 'adjacency_dict', 'train_mask', 'test_mask'])
edges = np.loadtxt("DBLP\edges.txt", dtype=int)
y = np.loadtxt("DBLP\labels.txt", dtype=int)

adjacency_dict = {}

for i in range(4057):
    adjacency_dict[i] = []

for array in edges:
    adjacency_dict[array[0]].append(array[1])

x = []
for i in range(4057):
    x.append([1])

# train_label = []
yt = []

for label in y:
    # x[label[0]][label[1]] = 1
    # train_label.append(label[1])
    yt.append(label[1])

train_mask = []
test_mask = []

for i in range(640):
    train_mask.append(True)
    test_mask.append(False)

for i in range(641, 800):
    train_mask.append(False)
    test_mask.append(True)

data = Data(x=np.array(x), y=np.array(yt), adjacency_dict=adjacency_dict,
                train_mask=np.array(train_mask), test_mask=np.array(test_mask))