import torch
import pickle
from torch.nn import Module, LeakyReLU, Dropout, Conv1d, Flatten, Linear
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
from memory import RehearsalMemory
import herding
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
np.ndarray.astype

def getSubset(X_train, y_train, indices, breakdown):
    subset = []
    sum = 0
    for group in breakdown:
        for i in range(group):
            sum = sum + indices[i]
        subset.append((X_train[:sum,:,:], y_train[:sum]))
        X_train = X_train[sum:,:,:]
        y_train = y_train[sum:]
        indices = indices[group:]
        sum=0
    return subset

def getFeatures(X_train, y_train, indices, y):
    if isinstance(indices, int):
        indices = [indices]
    num_samples = len(indices)
    X_temp = np.array([]) 
    y_temp = np.array([])
    t_temp = np.array([])
    z_temp = np.array([])
    if num_samples > 100:
        num_samples = 100
    if num_samples > 0 :
        X_temp = X_train[indices[:num_samples], :]
        y_temp = y_train[indices[:num_samples]]
        t_temp = np.array(indices[:num_samples])
        z_temp = np.array([y[i] for i in indices[:num_samples] if i < len(y)])

    return X_temp, y_temp, t_temp, z_temp

    
def normalize_input(x):
    m = np.mean(x, axis=1, keepdims=True)
    X = x - m
    return X

class GlassesNet(Module):

    def __init__(self):

        super(GlassesNet, self).__init__()
        self.lrelu = LeakyReLU()
        self.dropout1 = Dropout(0)
        self.dropout2 = Dropout(0)
        self.dropout3 = Dropout(0)
        self.dropout4 = Dropout(0.25)
        self.layer1 = Conv1d(7, 8, 7, 8)
        self.layer2 = Conv1d(8, 16, 5, 2)
        self.layer3 = Conv1d(16, 16, 3, 2)
        self.layer4 = Conv1d(16, 16, 3, 2)
        self.flatten = Flatten()
        self.fc = Linear(32, 16)

    def forward(self, x):
        y = self.dropout1(self.lrelu(self.layer1(x)))
        y = self.dropout2(self.lrelu(self.layer2(y)))
        y = self.dropout3(self.lrelu(self.layer3(y)))
        y = self.dropout4(self.lrelu(self.layer4(y)))
        y = self.flatten(y)
        y = self.fc(y)
        return y
    
class SimpleDataset(Dataset):

    def __init__(self, x, y):

        super(SimpleDataset, self).__init__()
        self.x = torch.tensor(x).permute(0, 2, 1).float()
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ind):
        return self.x[ind], self.y[ind]

with open("/..path/male#1.pickle", "rb") as f:
    tmp_data = pickle.load(f)
    tmp_data["walk_male#1"] = tmp_data.pop("walk")
    data = dict(tmp_data)

with open("/..path/male#2.pickle", "rb") as f:
    tmp_data = pickle.load(f)
    tmp_data["walk_male#2"] = tmp_data.pop("walk")
    data.update(tmp_data)

with open("/..path/male#3.pickle", "rb") as f:
    tmp_data = pickle.load(f)
    tmp_data["walk_male#3"] = tmp_data.pop("walk")
    data.update(tmp_data)

with open("/..path/male#4.pickle", "rb") as f:
    tmp_data = pickle.load(f)
    tmp_data["walk_male#4"] = tmp_data.pop("walk")
    data.update(tmp_data)

with open("/..path/female#1.pickle", "rb") as f:
    tmp_data = pickle.load(f)
    tmp_data["walk_female#1"] = tmp_data.pop("walk")
    data.update(tmp_data)

with open("/..path/female#2.pickle", "rb") as f:
    tmp_data = pickle.load(f)
    tmp_data["walk_female#2"] = tmp_data.pop("walk")
    data.update(tmp_data)

with open("/..path/female#3.pickle", "rb") as f:
    tmp_data = pickle.load(f)
    tmp_data["walk_female#3"] = tmp_data.pop("walk")
    data.update(tmp_data)

with open("/..path/female#4.pickle", "rb") as f:
    tmp_data = pickle.load(f)
    tmp_data["walk_female#4"] = tmp_data.pop("walk")
    data.update(tmp_data)


for key in data:
    if key != "test":
        data[key] = np.array(data[key])
print(data.keys())
print(data["walk_male#1"].shape, data["walk_male#2"].shape, data["walk_male#3"].shape, data["walk_male#4"].shape, data["walk_female#1"].shape, 
      data["walk_female#2"].shape, data["walk_female#3"].shape, data["walk_female#4"].shape)
X = np.concatenate([data["walk_male#1"], data["walk_male#2"], data["walk_male#3"], data["walk_male#4"], data["walk_female#1"], 
                    data["walk_female#2"],data["walk_female#3"],  data["walk_female#4"]], axis=0)
X = normalize_input(X)
y = np.concatenate([
    np.zeros((len(data["walk_male#1"]), 1)),
    np.ones((len(data["walk_male#2"]), 1)),
    np.ones((len(data["walk_female#1"]), 1)) * 2,
    np.ones((len(data["walk_female#2"]), 1)) * 3,
    np.ones((len(data["walk_male#3"]), 1)) * 4,
    np.ones((len(data["walk_female#3"]), 1)) * 5,
    np.ones((len(data["walk_male#4"]), 1)) * 6,
    np.ones((len(data["walk_female#4"]), 1)) * 7
], axis=0).astype(np.int64)
# write how you want to break you data [3,3,2], [3,5,0], etc
BATCH_ORDER = [3,3,2]
# per-person sample counts, in the same order X was concatenated above
indices = [
    len(data["walk_male#1"]), len(data["walk_male#2"]),
    len(data["walk_male#3"]), len(data["walk_male#4"]),
    len(data["walk_female#1"]), len(data["walk_female#2"]),
    len(data["walk_female#3"]), len(data["walk_female#4"]),
]
subset = getSubset(X, y, indices,BATCH_ORDER)
batch = 1
 
for i, (X_subset, y_subset) in enumerate(subset):
    
    print(f"\n STARTING BATCH {batch} WITH {BATCH_ORDER[batch-1]} SETS OF DATA ... \n")
   
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y_subset, test_size=0.1, shuffle=True, stratify=y_subset, random_state=seed)
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    model = GlassesNet()
    model.to(device)

    epochs = 30
    optimizer = Adam(model.parameters())
    loss_fn = CrossEntropyLoss(weight=torch.tensor([0.861, 0.866, 0.921, 0.861, 0.962, 0.846, 0.818, 0.864,
    0.861, 0.866, 0.921, 0.861, 0.962, 0.846, 0.818, 0.864]))

    train_acc = []
    val_acc = []
    best_predictions = None
    best_loss = np.inf
    best_model = None

    memory = RehearsalMemory(
    memory_size= 800,
    herding_method="barycenter",
    fixed_memory= True,
    nb_total_classes = 8,
    )
    memory.load("memory.npz")
    for X_train, y_train in subset:
        if batch >=1:
            mem_x, mem_y, mem_t = memory.get()
            mem_y = mem_y[:, None]  # Expand to 2D
            X_train = np.concatenate((X_train, mem_x), axis=0)
            y_train = np.concatenate((y_train, mem_y), axis=0)
        
    dataset = SimpleDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=3072, shuffle=True, num_workers=0)
    
        
    for epoch in range(epochs):
        model.train()
        train_acc.append(0)
    
        print("Epoch:", epoch)

        for X_tr, y_tr in dataloader:
            X_tr = X_tr.to(device)
            y_tr = y_tr.to(device)
            y_pred = model(X_tr)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, y_tr)
            loss.backward()
            optimizer.step()
            train_acc[-1] += torch.sum((torch.argmax(y_pred, axis=1) == y_tr).float()).cpu().numpy()
        train_acc[-1] /= len(dataset)
        model.eval()
        y_val = model(torch.tensor(X_test).permute(0, 2, 1).to(device).float())
        accuracy = torch.mean((torch.argmax(y_val, axis=1) == torch.tensor(y_test).to(device)).float()).cpu().numpy()
        loss = loss_fn(y_val.cpu(), torch.tensor(y_test)).detach().numpy()
        print("Accuracy", accuracy, "Loss", loss)
        val_acc.append(accuracy)
        if loss < best_loss:
            best_loss = loss
            best_predictions = torch.argmax(y_val, axis=1).cpu().numpy()
            best_model = model.state_dict()

    """dataloader = DataLoader(dataset, shuffle=False, num_workers=0)
    if batch == len(BATCH_ORDER):     
        X_temp, y_temp, t_temp, z_temp = getFeatures(X_train, y_train, indices,  y) 
        x = X_temp.astype(int)
        y = y_temp.astype(int)
        t = t_temp.astype(int)
        z = z_temp

        memory.add(x, y, t, z)
        memory.save("memory.npz")"""

    model.load_state_dict(best_model)
    model = model.cpu()
    model.eval()
    cm = confusion_matrix(y_test, best_predictions, normalize="pred")
    ax = sns.heatmap(cm, annot=True, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.show()
    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="validation")
    plt.legend()
    plt.show()

    test = data["test"]
    for walk in test:
        walk = normalize_input(walk)
        walk = torch.tensor(walk).float().permute(0, 2, 1)
        c_test = []
        prediction = model(walk)
        for label in torch.argmax(prediction, axis=1).reshape(-1):
            c_test.append(label.item())
        plt.scatter(np.arange(len(prediction)), torch.argmax(prediction, axis=1), c = c_test)
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ["walk_male#1", "walk_male#2", "walk_female#1", "walk_female#2","walk_male#3", "walk_female#3", "walk_male#4", "walk_female#4"])
        plt.show()
        prediction = torch.nn.functional.softmax(prediction, dim=1).T.reshape(1, 8, -1)
        prediction = torch.nn.ReflectionPad1d(2)(prediction)
        prediction = torch.nn.functional.conv1d(prediction, torch.ones((8, 1, 5))/5, padding="valid", groups=8).detach().cpu().numpy()[0].T
        c_test = []
        position = 0
        positions = []
        for label in np.argmax(prediction, axis=1).reshape(-1):
            position += (((label + 1) % 3) - 1)
            positions.append(position)
        plt.scatter(np.arange(len(prediction)), np.argmax(prediction, axis=1), c = c_test)
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ["walk_male#1", "walk_male#2", "walk_female#1", "walk_female#2","walk_male#3", "walk_female#3", "walk_male#4", "walk_female#4"])
        plt.figure()
        plt.plot(positions)
        plt.show()

    input("\n\n Press Enter to continue - this will clear the results and run the next batch")
    batch += 1
