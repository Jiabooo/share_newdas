import h5py
import torch
import shutil
from thop import profile
def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')         
        
def save_mask_checkpoint(state, mask_is_best,task_id, filename='maskcheckpoint.pth.tar'):
    torch.save(state, task_id+filename)
    if mask_is_best:
        shutil.copyfile(task_id+filename, task_id+'maskmodel_best.pth.tar')     
        
        
def cal_para(net):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        # print "stucture of layer: " + str(list(i.size()))
        for j in i.size():
            l *= j
        # print "para in this layer: " + str(l)
        k = k + l
    print("the amount of para: " + str(k))
    #inputs = torch.randn(1, 3, 768, 1024)
    inputs = torch.randn(1, 3, 589, 868)
    flops, params = profile(net, (inputs,))
    print('flops: ', flops, 'params: ', params)
    
    