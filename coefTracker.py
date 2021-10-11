import torch
import torch.nn as nn
import torch.nn.functional as F
class coefPredictNet_v1(nn.Module):
    def __init__(self):
        super(coefPredictNet_v1, self).__init__()
        
        self.predictFC = nn.Linear(32, 32)
        self.combineFC = nn.Linear(64, 32)
        
    def forward(self, coef_t0, coef_t1):
        p_coef_t1 = self.predictFC(coef_t0) #torch.Size([1,32])
        x = torch.cat((p_coef_t1, coef_t1), dim=1) #torch.Size([1,64])
        x = self.combineFC(x)
        return x
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

class coefPredictNet_v2(nn.Module):
    def __init__(self):
        super(coefPredictNet_v2, self).__init__()
        
        self.coef1featFC = nn.Linear(32, 4)
        self.coef2featFC = nn.Linear(32, 4)
        self.combineFC = nn.Linear(8, 32)
        
    def forward(self, coef_t0, coef_t1):
        f_coef_t0 = self.coef1featFC(coef_t0) #torch.Size([1,4])
        f_coef_t1 = self.coef2featFC(coef_t1) #torch.Size([1,4])
        x = torch.cat((f_coef_t0, f_coef_t1), dim=1) #torch.Size([1,8])
        x = self.combineFC(x) #torch.Size([1,32])
        x = coef_t1 + x #skip connection
        return x
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
            
class coefPredictNet_v3(nn.Module):
    '''
    predict next coefficients
    '''
    def __init__(self):
        super(coefPredictNet_v3, self).__init__()
        
        self.combineFC = nn.Linear(64, 32)
        self.predictFC = nn.Linear(32, 32)
        
    def forward(self, coef_t0, coef_t1):
        x = torch.cat((coef_t0, coef_t1), dim=1) #torch.Size([1,64])
        x = self.combineFC(x) #torch.Size([1,32])
        x = self.predictFC(x) #torch.Size([1,32])
        x = coef_t1 + x #skip connection
        return x
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

class coefPredictNet_v4(nn.Module):
    '''
    predict next coefficients
    '''
    def __init__(self):
        super(coefPredictNet_v4, self).__init__()
        
        #self.combineFC = nn.Linear(64, 32)
        self.predictFC = nn.Linear(32, 32)
        
    def forward(self, coef_t0, coef_t1):
        x = coef_t1 - coef_t0
        #x = self.combineFC(x) #torch.Size([1,32])
        x = self.predictFC(x) #torch.Size([1,32])
        x = coef_t1 + x #skip connection
        return x
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

class coefPredictNet_v5(nn.Module):
    '''
    predict next coefficients
    '''
    def __init__(self):
        super(coefPredictNet_v5, self).__init__()
        
        self.combineFC1 = nn.Linear(64, 8)
        self.combineFC2 = nn.Linear(64, 8)
        self.combineFC3 = nn.Linear(64, 8)
        self.combineFC4 = nn.Linear(64, 8)
        self.hiddenFC1 = nn.Linear(8, 8)
        self.hiddenFC2 = nn.Linear(8, 8)
        self.hiddenFC3 = nn.Linear(8, 8)
        self.hiddenFC4 = nn.Linear(8, 8)
        self.predictFC1 = nn.Linear(8, 32)
        self.predictFC2 = nn.Linear(8, 32)
        self.predictFC3 = nn.Linear(8, 32)
        self.predictFC4 = nn.Linear(8, 32)
        
    def forward(self, coef_t0, coef_t1):
        x = torch.cat((coef_t0, coef_t1), dim=1) #torch.Size([1,64])
        x1 = self.combineFC1(x) #torch.Size([1,32])
        x2 = self.combineFC2(x) #torch.Size([1,32])
        x3 = self.combineFC3(x) #torch.Size([1,32])
        x4 = self.combineFC4(x) #torch.Size([1,32])
        x1 = self.hiddenFC1(x1) #torch.Size([1,32])
        x2 = self.hiddenFC2(x2) #torch.Size([1,32])
        x3 = self.hiddenFC3(x3) #torch.Size([1,32])
        x4 = self.hiddenFC4(x4) #torch.Size([1,32])
        x1 = self.predictFC1(x1) #torch.Size([1,32])
        x2 = self.predictFC2(x2) #torch.Size([1,32])
        x3 = self.predictFC3(x3) #torch.Size([1,32])
        x4 = self.predictFC4(x4) #torch.Size([1,32])
        x = x1 + x2 + x3 + x4
        x = coef_t1 + x #skip connection
        return x
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
class coefPredictNet_v6(nn.Module):
    '''
    predict next coefficients
    '''
    def __init__(self):
        super(coefPredictNet_v6, self).__init__()
        
        self.predictFC1 = nn.Linear(32, 32)
        self.predictFC2 = nn.Linear(32, 32)
        
        self.combineFC1 = nn.Linear(64, 32)
        self.combineFC2 = nn.Linear(32, 32)
        
    def forward(self, coef_t0, coef_t1):
    
        x = self.predictFC1(coef_t0) #torch.Size([1,32])
        x = self.predictFC2(x) #torch.Size([1,32])
        x = coef_t0 + x #skip connection
        
        
        x = torch.cat((x, coef_t1), dim=1) #torch.Size([1,64])
        x = self.combineFC1(x) #torch.Size([1,32])
        x = self.combineFC2(x) #torch.Size([1,32])
        x = coef_t1 + x #skip connection
        return x
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
            
class coefPredictNet_v7(nn.Module):
    '''
    predict next coefficients
    '''
    def __init__(self):
        super(coefPredictNet_v7, self).__init__()
        
        self.LSTM = nn.LSTM(input_size=64, hidden_size=32, num_layers=3)
        self.previous_proj = None
        
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(3,batch_size,32).detach()
        cell_state = torch.zeros(3,batch_size,32).detach()
        self.previous_proj = (hidden_state, cell_state)
        return
        
    def forward(self, coef_t0, coef_t1):
        x = torch.cat((coef_t0, coef_t1), dim=1) #torch.Size([1,64])
        x = x.unsqueeze(0) #torch.Size([1,1,64])
        out, proj = self.LSTM(x, self.previous_proj) #torch.Size([1,1,32])
        self.previous_proj = (proj[0].detach(), proj[1].detach())
        return out.squeeze(0)
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
            
            
class coefPredictNet_v8(nn.Module):
    '''
    predict next coefficients
    '''
    def __init__(self):
        super(coefPredictNet_v8, self).__init__()
        
        self.LSTM = nn.LSTM(input_size=64, hidden_size=32, num_layers=4)
        self.previous_proj = None
        
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(4,batch_size,32).detach()
        cell_state = torch.zeros(4,batch_size,32).detach()
        self.previous_proj = (hidden_state, cell_state)
        return
        
    def forward(self, coef_t0, coef_t1):
        x = torch.cat((coef_t0, coef_t1), dim=1) #torch.Size([1,64])
        x = x.unsqueeze(0) #torch.Size([1,1,64])
        out, proj = self.LSTM(x, self.previous_proj) #torch.Size([1,1,32])
        self.previous_proj = (proj[0].detach(), proj[1].detach())
        return out.squeeze(0)
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')

class coefPredictNet_v9(nn.Module):
    '''
    predict next coefficients
    '''
    def __init__(self):
        super(coefPredictNet_v9, self).__init__()
        
        self.LSTM = nn.LSTM(input_size=64, hidden_size=32, num_layers=5)
        self.previous_proj = None
        
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(5,batch_size,32).detach()
        cell_state = torch.zeros(5,batch_size,32).detach()
        self.previous_proj = (hidden_state, cell_state)
        return
        
    def forward(self, coef_t0, coef_t1):
        x = torch.cat((coef_t0, coef_t1), dim=1) #torch.Size([1,64])
        x = x.unsqueeze(0) #torch.Size([1,1,64])
        out, proj = self.LSTM(x, self.previous_proj) #torch.Size([1,1,32])
        self.previous_proj = (proj[0].detach(), proj[1].detach())
        return out.squeeze(0)
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')
            
class coefPredictNet_v10(nn.Module):
    '''
    predict refine coefficients
    '''
    def __init__(self):
        super(coefPredictNet_v10, self).__init__()
        
        self.LSTM = nn.LSTM(input_size=64, hidden_size=34, num_layers=5)
        self.previous_proj = None
        
    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(5,batch_size,34).detach()
        cell_state = torch.zeros(5,batch_size,34).detach()
        self.previous_proj = (hidden_state, cell_state)
        return
        
    def forward(self, coef_t0, coef_t1):
        x = torch.cat((coef_t0, coef_t1), dim=1) #torch.Size([1,64])
        x = x.unsqueeze(0) #torch.Size([1,1,64])
        out, proj = self.LSTM(x, self.previous_proj) #torch.Size([1,1,32])
        self.previous_proj = (proj[0].detach(), proj[1].detach())
        return out.squeeze(0)
        
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
        
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)
        try:
            self.load_state_dict(state_dict)
        except RuntimeError as e:
            print('Ignoring "' + str(e) + '"')