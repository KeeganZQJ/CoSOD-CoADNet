from common_packages import *
from misc import *


class List_Loader(data.Dataset):
    def __init__(self, data_root, list_file, is_augment):
        self.data_root = data_root
        self.list_file = list_file
        self.is_augment = is_augment
        self.load_path = [os.path.join(data_root, line.strip()) for line in open(os.path.join(data_root, list_file), 'r')]
        self.to_tensor = transforms.ToTensor()
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1)
    def __getitem__(self, index):
        path = self.load_path[index]
        prefix = path.split('/')[-1]
        image = Image.open(os.path.join(path + '.jpg'))
        label = Image.open(os.path.join(path + '.png'))
        if self.is_augment:
            if np.random.random() > 0.50:
                image = self.horizontal_flip(image)
                label = self.horizontal_flip(label)
            image, label = random_crop_image_label(image, label, expansion_ratio=np.random.random()*0.20+0.02)
            ps = int(random.choice(10)) # zero-padding size
            zero_pad = nn.ZeroPad2d(ps)
            image = zero_pad(image[:, ps:(image.size(1)-ps), ps:(image.size(2)-ps)])
            label = zero_pad(label[:, ps:(label.size(1)-ps), ps:(label.size(2)-ps)])
        else:
            image = self.to_tensor(image)
            label = self.to_tensor(label)
        return image, label, prefix
    def __len__(self):
        return len(self.load_path)

    
def SOD_Loader(load_path_list, batch_size, is_augment):
    # image_tensor: # [batch_size, 3, H, W]
    # label_tensor: # [batch_size, 3, H, W]
    # prefix: a list containing the file prefixes
    num_total = len(load_path_list)
    indices = random.choice(num_total, batch_size, replace=False)
    image_tensor = []
    label_tensor = []
    prefix = []
    for path_id in indices:
        load_path = load_path_list[path_id]
        prefix.append(load_path.split('/')[-1])
        image = Image.open(load_path + '.jpg')
        label = Image.open(load_path + '.png')
        if is_augment:
            horizontal_flip = transforms.RandomHorizontalFlip(p=1)
            if np.random.random() > 0.5:
                image = horizontal_flip(image)
                label = horizontal_flip(label)
            image, label = random_crop_image_label(image, label, expansion_ratio=np.random.random()*0.20+0.02)
            ps = int(random.choice(10)) # zero-padding size
            zero_pad = nn.ZeroPad2d(ps)
            image = zero_pad(image[:, ps:(image.size(1)-ps), ps:(image.size(2)-ps)])
            label = zero_pad(label[:, ps:(label.size(1)-ps), ps:(label.size(2)-ps)])
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
            label = to_tensor(label)
        image_tensor.append(image.unsqueeze(0))
        label_tensor.append(label.unsqueeze(0))
    image_tensor = torch.cat(image_tensor, dim=0) # [batch_size, 3, H, W]
    label_tensor = torch.cat(label_tensor, dim=0) # [batch_size, 3, H, W]
    return image_tensor, label_tensor, prefix


def Group_Stat_Parser(data_root, stat_file, list_file):
    # group_numb: number of different groups
    # group_name: name of each group
    # group_size: the i^th group contains group_size[i] samples
    # group_list: group_list[i] is a list, which contains loading paths of all group_size[i] samples in the i^th group
    group_stats = [line.strip().split(' ') for line in open(os.path.join(data_root, stat_file), 'r')]
    group_numb = len(group_stats)
    group_name = []
    group_size = []
    for item in group_stats:
        group_name.append(item[0])
        group_size.append(int(item[1]))
    path_list = [os.path.join(data_root, line.strip()) for line in open(os.path.join(data_root, list_file), 'r')]
    num_accumulated = []
    for i in range(group_numb):
        temp = 0
        for j in range(i + 1):
            temp += group_size[j]
        num_accumulated.append(temp)
    group_list = []
    for grp_id in range(group_numb):
        if grp_id == 0:
            start_index = 0
            end_index = num_accumulated[0]
        else:
            start_index = num_accumulated[grp_id-1]
            end_index = num_accumulated[grp_id]
        this_list = []
        for idx in range(start_index, end_index):
            this_list.append(path_list[idx])
        group_list.append(this_list)
    return group_numb, group_name, group_size, group_list


def supp(group_size, group_list, mini_group_size=5):
    assert len(group_size) == len(group_list)
    num_groups = len(group_size)
    for gid in range(num_groups):
        gs = group_size[gid]
        if gs < mini_group_size:
            # print('add [{}] sample(s) to group [{}]'.format(mini_group_size-gs, align_number(gid+1, 3)))
            # add samples to the corresponding group_list
            for supp_id in range(0, mini_group_size-gs):
                group_list[gid].append(group_list[gid][supp_id])
            # do not forget to accordingly update group_size
            group_size[gid] += (mini_group_size-gs)
    return group_size, group_list


def CoSOD_Loader(group_numb, group_list, B, M, is_augment):
    # batch_group_images: tensor, [B, M, 3, H, W]
    # batch_group_labels: tensor, [B, M, 1, H, W]
    # batch_group_prefixes: batch_group_prefixes[b][m] provides the file prefix of the m^th input data in the b^th batch
    batch_group_images = []
    batch_group_labels = []
    batch_group_prefixes = []
    group_indices = random.choice(group_numb, B, replace=False) # select B different groups (replace=False)
    for gid in group_indices:
        group_image_indices = random.choice(len(group_list[gid]), M, replace=False) # select M different images from each of the B groups
        group_images = []
        group_labels = []
        group_prefixes = []
        for iid in group_image_indices:
            load_path = group_list[gid][iid]
            image = Image.open(load_path + '.jpg')
            label = Image.open(load_path + '.png')
            if is_augment:
                horizontal_flip = transforms.RandomHorizontalFlip(p=1)
                if np.random.random() > 0.5:
                    image = horizontal_flip(image)
                    label = horizontal_flip(label)
                image, label = random_crop_image_label(image, label, expansion_ratio=np.random.random()*0.20+0.02)
                ps = int(random.choice(10)) # zero-padding size
                zero_pad = nn.ZeroPad2d(ps)
                image = zero_pad(image[:, ps:(image.size(1)-ps), ps:(image.size(2)-ps)])
                label = zero_pad(label[:, ps:(label.size(1)-ps), ps:(label.size(2)-ps)])
            else:
                to_tensor = transforms.ToTensor()
                image = to_tensor(image)
                label = to_tensor(label)
            group_images.append(image.unsqueeze(0))
            group_labels.append(label.unsqueeze(0))
            group_prefixes.append(load_path.split('/')[-1])
        group_images = torch.cat(group_images, dim=0).unsqueeze(0) # [1, M, 3, H, W]
        group_labels = torch.cat(group_labels, dim=0).unsqueeze(0) # [1, M, 1, H, W]
        batch_group_images.append(group_images)
        batch_group_labels.append(group_labels)
        batch_group_prefixes.append(group_prefixes)
    batch_group_images = torch.cat(batch_group_images, dim=0) # [B, M, 3, H, W]
    batch_group_labels = torch.cat(batch_group_labels, dim=0) # [B, M, 1, H, W]
    return batch_group_images, batch_group_labels, batch_group_prefixes



def Identity_Loader(load_path_list, Bg, Bs):
    to_tensor = transforms.ToTensor()
    horizontal_flip = transforms.RandomHorizontalFlip(p=1)
    num_total = len(load_path_list)
    indices_g = random.choice(num_total, Bg, replace=False)
    # load CoSOD data
    gi_loaded = []
    gl_loaded = []
    for idx in indices_g:
        load_path = load_path_list[idx]
        I_1 = Image.open(os.path.join(load_path + '.jpg'))
        L_1 = Image.open(os.path.join(load_path + '.png'))
        H, W = I_1.size
        I_2 = horizontal_flip(I_1)
        L_2 = horizontal_flip(L_1)
        I_3, L_3 = random_crop_image_label(I_1, L_1, expansion_ratio=np.random.random()*0.20+0.02)
        I_4, L_4 = random_crop_image_label(I_2, L_2, expansion_ratio=np.random.random()*0.20+0.02)
        ct_crop_ratio = (1.0 - np.random.random()*0.20)
        ct_crop_h, ct_crop_w = int(H*ct_crop_ratio), int(W*ct_crop_ratio)
        center_crop = transforms.CenterCrop(size=(ct_crop_h, ct_crop_w))
        if np.random.random() > 0.50:
            I_5, L_5 = center_crop(I_1).resize((H, W)), center_crop(L_1).resize((H, W))
        else:
            I_5, L_5 = center_crop(I_2).resize((H, W)), center_crop(L_2).resize((H, W))
        I_1, L_1 = to_tensor(I_1), to_tensor(L_1)
        I_2, L_2 = to_tensor(I_2), to_tensor(L_2)
        I_5, L_5 = to_tensor(I_5), to_tensor(L_5)
        ps = int(random.choice(10)) # zero-padding size
        zero_pad = nn.ZeroPad2d(ps)
        gi = torch.cat((I_1.unsqueeze(0), I_2.unsqueeze(0), I_3.unsqueeze(0), I_4.unsqueeze(0), I_5.unsqueeze(0)), dim=0)
        gl = torch.cat((L_1.unsqueeze(0), L_2.unsqueeze(0), L_3.unsqueeze(0), L_4.unsqueeze(0), L_5.unsqueeze(0)), dim=0)
        gi = zero_pad(gi[:, :, ps:(H-ps), ps:(W-ps)]) # [5, 3, H, W]
        gl = zero_pad(gl[:, :, ps:(H-ps), ps:(W-ps)]) # [5, 1, H, W]
        gl = binarize_label(gl, 0.50)
        gi_loaded.append(gi.unsqueeze(0))
        gl_loaded.append(gl.unsqueeze(0))
    gi_loaded = torch.cat(gi_loaded, dim=0) # [Bg, 5, 3, H, W]
    gl_loaded = torch.cat(gl_loaded, dim=0) # [Bg, 5, 3, H, W]
    # load SOD data
    si_loaded, sl_loaded, _ = SOD_Loader(load_path_list, Bs, is_augment=True) # [Bs, 3, H, W], [Bs, 3, H, W]
    # gi_loaded: [Bg, 5, 3, H, W]
    # gl_loaded: [Bg, 5, 1, H, W]
    # si_loaded: [Bs, 3, H, W]
    # sl_loaded: [Bs, 1, H, W]
    return [gi_loaded, gl_loaded], [si_loaded, sl_loaded]
    
    
    