from backbone import Backbone_Wrapper_VGG16, Backbone_Wrapper_ResNet50, Backbone_Wrapper_Dilated_ResNet50
from common_packages import *
from dataset import *
from modules import *
from network import *
from misc import *
from ops import *
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
gpu_devices = list(np.arange(torch.cuda.device_count()))
num_gpu = len(gpu_devices)
multi_gpu = num_gpu > 1
from torch.utils.data.distributed import DistributedSampler
torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
ckpt_root = './ckpt'
data_root = '../auxi'
list_file = 'List-DUTS-TR.txt'
load_path_list = [os.path.join(data_root, line.strip()) for line in open(os.path.join(data_root, list_file), 'r')]


net = CoADNet_Dilated_ResNet50(mode='train', compute_loss=True).cuda()
if torch.distributed.get_rank() == 0:
    net.backbone.load_state_dict(torch.load(os.path.join(ckpt_root, 'pretrained', 'Backbone_DRN50_COCO-SEG.pth')))
if torch.cuda.device_count() > 1:
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
    
    
Bg = 4
Bs = 8
M = 5
iterations = 50000
show_every = 1000
save_every = 1000
init_lr = 1e-4
min_lr = 1e-6

bb_params = list(map(id, net.module.backbone.parameters()))
ot_params = filter(lambda p: id(p) not in bb_params, net.parameters())
params = [{'params': net.module.backbone.parameters(), 'lr': init_lr * 0.01}, {'params': ot_params, 'lr': init_lr * 1.0}]
optimizer = torch.optim.Adam(params, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=min_lr)


record_c_loss = 0
record_s_loss = 0
num_accum_c = 0
num_accum_s = 0
net.train()
for it in tqdm.tqdm(range(1, iterations + 1)):
    optimizer.zero_grad()
    [gi, gl], [si, sl] = Identity_Loader(load_path_list, Bg, Bs)
    gi, gl, si, sl = gi.cuda(), gl.cuda(), si.cuda(), sl.cuda()
    csm, c_loss, s_loss = net(gi, si, gl, sl)
    loss = c_loss + s_loss
    record_c_loss += c_loss.item() * (Bg*M)
    record_s_loss += s_loss.item() * Bs
    num_accum_c += Bg*M
    num_accum_s += Bs
    loss.backward()
    optimizer.step()
    scheduler.step()
    if np.mod(it, show_every) == 0:
        record_c_loss = np.around(record_c_loss/num_accum_c, 5)
        record_s_loss = np.around(record_s_loss/num_accum_s, 5)
        if torch.distributed.get_rank() == 0:
            print('iteration: {}, c_loss: {}, s_loss: {}'.format(align_number(it, 5), record_c_loss, record_s_loss))
        record_c_loss = 0
        record_s_loss = 0
        num_accum_c = 0
        num_accum_s = 0
    if np.mod(it, save_every) == 0:
        if torch.distributed.get_rank() == 0:
            torch.save(net.module.state_dict(), os.path.join(ckpt_root, 'pretrained', 'CoADNet_DRN50_COCO-SEG_Pretrained.pth'))
        
        