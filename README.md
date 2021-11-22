# Online Multiple Object Segmentation in Mask Coefxcient Space
 This repo is a implementation of [my master's thesis](https://hdl.handle.net/11296/7s4pwc)
 
 In my thesis, I use yolact model with some tracking algorithm(like Deep SORT) to achieve online instance segmentation.
 Therefore, most of the code in this repo(to implement YOLACT) is clone from [dbolya/yolact](https://github.com/dbolya/yolact).Thanks for their contribution!
 
 And also, [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0) is required to implement YOLACT++ in original repo from Daniel Bolya.
 But the GPU in my hand is only one RTX 3080, and it only support CUDA 11 up ;)
 Consider all supporting problem about runnning environment(versions about python, pytorch, cuda, GPU)
 I use:
        GPU      :RTX 3080
        Python   :verion 3.8
        PyTorch  :1.7.0
        CUDA     :11.0
        cudnn    :8.0
 in this repository.
 If you have similar environment problem to me, you might change to this [DCN verion](https://github.com/jinfagang/DCNv2_latest)
 
 Last, some of the code in this repo to implement tracking algorithm be like Deep SORT is refer to [ifzhang/FairMOT](https://github.com/ifzhang/FairMOT).
 I have learned a lot from it. Thanks to their contributions too!
