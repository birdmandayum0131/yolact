import torch
import torch.nn as nn
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