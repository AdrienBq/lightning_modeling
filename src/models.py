import os
import sys
from typing import List
import math
import torch
import torch.nn as nn
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
import pickle

# Path to the config file (relative or absolute)
CONFIG_FILE = Path("config.json")

# Load config
with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

# Use the paths
PAR_DIR = config["PAR_DIR"]
if PAR_DIR not in sys.path:
    sys.path.append(PAR_DIR)


class BaselineModel(nn.Module):
    def __init__(self, maps_dic: dict, name: str = "climatology_baseline", **kwargs):
        super().__init__()
        self.maps = nn.ParameterDict({season: nn.Parameter(maps_dic[season], requires_grad=False) for season in maps_dic.keys()})
        self.name = name
        self.save_path = kwargs.get("save_path", None)

    def forward(self, x, season=None):
        # ignore input, always return the same map
        batch_size = x.shape[0]
        return self.maps[season].expand(batch_size, *self.maps[season].shape)
    
    def save(self):
        if self.save_path is None:
            raise ValueError("save_path is not defined")
        with open(self.save_path / f"{self.name}.pkl", "wb") as f:            
            pickle.dump(self.maps, f)


class LogisticRegressionModel(nn.Module):
    def __init__(self, model, name: str = "logistic_regression", remove_vars: List[int] = None, **kwargs):
        super().__init__()
        self.model = model
        self.name = name
        self.remove_vars = remove_vars
        self.save_path = kwargs.get("save_path", None)

    def forward(self, x):
        if self.remove_vars is not None and len(self.remove_vars) > 0:
            if len(self.remove_vars) == 5:
                x = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
            elif len(self.remove_vars) < 5:
                remove = torch.tensor(self.remove_vars, device=x.device)
                keep = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
                keep[remove] = False
                x = x[:, keep, :, :]
            else :
                raise ValueError("removed_features should be a list of integers between 0 and 4")
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        y_pred_flat = self.model.predict_proba(x_flat)[:, 1]
        y_pred = y_pred_flat.reshape(x.shape[0], x.shape[2], x.shape[3])
        return torch.tensor(y_pred).float()
    
    def save(self):
        if self.save_path is None:
            raise ValueError("save_path is not defined")
        with open(self.save_path / f"{self.name}.pkl", "wb") as f:            
            pickle.dump(self.model, f)


class XGBoostModel(nn.Module):
    def __init__(self, model, name: str = "xgboost_model", remove_vars: List[int] = None, **kwargs):
        super().__init__()
        self.model = model
        self.name = name
        self.remove_vars = remove_vars
        self.save_path = kwargs.get("save_path", None)

    def forward(self, x):
        if self.remove_vars is not None and len(self.remove_vars) > 0:
            if len(self.remove_vars) == 5:
                x = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
            elif len(self.remove_vars) < 5:
                remove = torch.tensor(self.remove_vars, device=x.device)
                keep = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
                keep[remove] = False
                x = x[:, keep, :, :]
            else :
                raise ValueError("removed_features should be a list of integers between 0 and 4")
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        y_pred_flat = self.model.predict_proba(x_flat.numpy())[:, 1]
        y_pred = y_pred_flat.reshape(x.shape[0], x.shape[2], x.shape[3])
        return torch.tensor(y_pred).float()
    
    def save(self):
        if self.save_path is None:
            raise ValueError("save_path is not defined")
        with open(self.save_path / f"{self.name}.pkl", "wb") as f:            
            pickle.dump(self.model, f)
    

class GAMModel(nn.Module):
    def __init__(self, model, name: str = "gam_model", remove_vars: List[int] = None, **kwargs):
        super().__init__()
        self.model = model
        self.name = name
        self.remove_vars = remove_vars
        self.save_path = kwargs.get("save_path", None)

    def forward(self, x):
        if self.remove_vars is not None and len(self.remove_vars) > 0:
            if len(self.remove_vars) == 5:
                x = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
            elif len(self.remove_vars) < 5:
                remove = torch.tensor(self.remove_vars, device=x.device)
                keep = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
                keep[remove] = False
                x = x[:, keep, :, :]
            else :
                raise ValueError("removed_features should be a list of integers between 0 and 4")
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        y_pred_flat = self.model.predict_proba(x_flat.numpy())
        y_pred = y_pred_flat.reshape(x.shape[0], x.shape[2], x.shape[3])
        return torch.tensor(y_pred).float()
    
    def save(self):
        if self.save_path is None:
            raise ValueError("save_path is not defined")
        with open(self.save_path / f"{self.name}.pkl", "wb") as f:            
            pickle.dump(self.model, f)