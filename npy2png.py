import numpy as np
from PIL import Image
import os
imageSize = (480,480)
dir="./scripts/"#npy檔案路徑
dest_dir="./prototypes/motorbike/"
def npy2png(dir,dest_dir):
    if os.path.exists(dir)==False:
        os.makedirs(dir)
    if os.path.exists(dest_dir)==False:
        os.makedirs(dest_dir)
    file=dir+'proto.npy'
    prototypes=np.load(file)
    prototypes = prototypes*255
    
    count=0
    for idx in range(prototypes.shape[2]):
        proto = prototypes[:,:,idx:idx+1]
        width=proto.shape[0]
        height=proto.shape[1]

        proto=np.reshape(proto,(width,height))
        grayscale_proto=Image.fromarray(proto).convert("L")
        grayscale_proto = grayscale_proto.resize(imageSize)
        grayscale_proto.save(dest_dir+"proto_"+str(idx)+".png")

if __name__=="__main__":
    npy2png(dir,dest_dir)