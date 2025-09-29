import torch 
import torch.nn as nn
import joblib
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from torch.utils.data import random_split
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
scale = MinMaxScaler()
target = MinMaxScaler()

class csv_dataset(Dataset):
    def __init__(self,csv_filepath):
        super().__init__()
        df = pd.read_csv(csv_filepath)
        
        colu = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition','Fertilizer_Used','Irrigation_Used']
        numeric_cols = ["Rainfall_mm", "Temperature_Celsius"]
        enc_colu = encoder.fit_transform(df[colu])
        enc_df = pd.DataFrame(enc_colu,columns=encoder.get_feature_names_out(colu))

        feature_df = pd.concat([enc_df, df[numeric_cols]], axis=1)
        target_df = pd.DataFrame(df['Yield_tons_per_hectare'])

        norm_features = scale.fit_transform(feature_df)
        norm_target = target.fit_transform(target_df)

        joblib.dump(encoder,"onehotencoder.pkl")
        joblib.dump(scale,"minmaxscale1.pkl")
        joblib.dump(target,"minmaxscale2.pkl")

        self.X = torch.tensor(norm_features.astype(np.float32))
        self.y = torch.tensor(norm_target.astype(np.float32))

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]

class linear_reg(nn.Module):
    def __init__(self,input,output):
        super().__init__()

        self.linear1 = nn.Linear(input, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, output)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.l_relu = nn.LeakyReLU()

    def forward(self,x):
        x = self.l_relu(self.bn1(self.linear1(x)))
        x = self.l_relu(self.bn2(self.linear2(x)))
        x = self.l_relu(self.bn3(self.linear3(x)))
        x = self.linear4(x)
        return x

def check_accuracy(model,loader,device = "cuda" if torch.cuda.is_available() else "cpu"):
    model.load_state_dict(torch.load("model1.pth"))
    model = model.to(device)
    model.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            output = model(x)
            all_preds.append(output.view(-1).cpu())
            all_targets.append(y.view(-1).cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    score = r2_score(all_targets, all_preds)
    return score

def train(model,train_dataloader,epochs=20,lr=0.001,device = "cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
    for epoch in range(epochs):
        for x,y in train_dataloader:
            x,y = x.to(device),y.squeeze().to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
        print(f"epoch:{epoch},loss:{loss}")
    torch.save(model.state_dict(),"model1.pth")
    return model

if __name__ == "__main__":
    data = csv_dataset(r'C:\Users\chara\Desktop\Desktop\vs code\langchain\sih\crop_yield.csv')
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])

    train_dataloader = DataLoader(dataset=train_data,shuffle=True,batch_size=128)
    test_dataloader = DataLoader(dataset=test_data,shuffle=True,batch_size=128)
    model = linear_reg(25,1)
    model = train(model,train_dataloader)
    print(check_accuracy(model=model,loader=test_dataloader))
    model.state_dict(torch.load("model1.pth"))
    encoder = joblib.load("onehotencoder.pkl")
    scale = joblib.load("minmaxscale2.pkl")
    
    for x,y in test_dataloader:
        data1 = x
        data_pred = y
        break
    print(data1)
    print(data_pred)
    y_pred = model(data1)
    print(y_pred)
    print(scale.inverse_transform(y_pred.detach().numpy()))
    print("---------------------------------------")
    print(scale.inverse_transform(data_pred.reshape(-1, 1).detach().numpy()))