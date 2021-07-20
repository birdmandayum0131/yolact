import os
save_folder = "CTweights/"
def train():
    '''建立存檔資料夾'''
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)