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
data_root = '../data'


M = 5
train_dataset = 'COCO-SEG'
auxil_dataset = 'DUTS-TR'
train_stat_file = 'Stat-' + train_dataset + '.txt'
train_list_file = 'List-' + train_dataset + '.txt'
group_numb, group_name, group_size, group_list = Group_Stat_Parser(data_root, train_stat_file, train_list_file)
group_size, group_list = supp(group_size, group_list, M)
auxil_list_file = 'List-' + auxil_dataset + '.txt'
auxil_load_path = [os.path.join(data_root, line.strip()) for line in open(os.path.join(data_root, auxil_list_file), 'r')]


net = CoADNet_Dilated_ResNet50(mode='train', compute_loss=True).cuda()
resume_training = False
if torch.distributed.get_rank() == 0:
    net.load_state_dict(torch.load(os.path.join(ckpt_root, 'pretrained', 'CoADNet_DRN50_COCO-SEG_Pretrained.pth')))
    if resume_training:
        print('activate resume training.')
        net.load_state_dict(torch.load('XXX.pth'))
if torch.cuda.device_count() > 1:
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
    
    
Bg = 4
Bs = 8
iterations = 50000
show_every = 1000
save_every = 1000
multi_ckpt = False
init_lr = 1e-4
min_lr = 1e-5


bb_params = list(map(id, net.module.backbone.parameters()))
ot_params = filter(lambda p: id(p) not in bb_params, net.parameters())
params = [{'params': net.module.backbone.parameters(), 'lr': init_lr * 0.05}, {'params': ot_params, 'lr': init_lr * 1.0}] # 0.01, 0.05
optimizer = torch.optim.Adam(params, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations, eta_min=min_lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 5000, gamma=0.75, last_epoch=-1)


record_c_loss = 0
record_s_loss = 0
num_accum_c = 0
num_accum_s = 0
net.train()
for it in tqdm.tqdm(range(1, iterations + 1)):
    optimizer.zero_grad()
    gi, gl, g_prefixes = CoSOD_Loader(group_numb, group_list, Bg, M, is_augment=True)
    si, sl, s_prefixes = SOD_Loader(auxil_load_path, Bs, is_augment=True)
    gi, gl, si, sl = gi.cuda(), gl.cuda(), si.cuda(), sl.cuda()
    csm, c_loss, s_loss = net(gi, si, gl, sl)
    loss = c_loss * 1.0 + s_loss * 0.1
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
        if multi_ckpt:
            if np.mod(it, save_every)==0:
                if torch.distributed.get_rank() == 0:
                    model_name = 'CoADNet_DRN50_COCO-SEG_' + align_number(it, 5) + '.pth'
                    torch.save(net.module.state_dict(), os.path.join(ckpt_root, 'trained', model_name))
if torch.distributed.get_rank() == 0:
    torch.save(net.module.state_dict(), os.path.join(ckpt_root, 'trained', 'CoADNet_DRN50_COCO-SEG.pth'))
    
    
    