import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.nn.functional as F
import pdb
import math
import argparse
import sys
sys.dont_write_bytecode = True







''' 

	@Article{chen2021multi,
	author  = {Chen, Haoxing and Li, Huaxiong and Li, Yaohui and Chen, Chunlin},
	title   = {Multi-level Metric Learning for Few-Shot Image Recognition},
	journal = {arXiv preprint arXiv:2103.11383},
	year    = {2021},
}

'''


###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_FewShotNet(pretrained=False, model_root=None, which_model='Conv64', norm='batch', init_type='normal',num_classes=5,
                      use_gpu=True, shot_num=5,query_num=15, **kwargs):
    FewShotNet = None
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model == 'Conv64F':
        #FewShotNet = Conv_64F(norm_layer=norm_layer, **kwargs)
        FewShotNet = FourLayer_64F(norm_layer=norm_layer, num_classes=num_classes, shot_num=shot_num, **kwargs)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % which_model)
    init_weights(FewShotNet, init_type=init_type)

    if use_gpu:
        FewShotNet.cuda()

    if pretrained:
        FewShotNet.load_state_dict(model_root)

    return FewShotNet


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)




class FourLayer_64F(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm1d, num_classes=5, neighbor_k=3, batch_size=4, shot_num=1,query_num = 15):
        super(FourLayer_64F, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm1d
        else:
            use_bias = norm_layer == nn.InstanceNorm1d

        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
        #     norm_layer(64),
        #     nn.LeakyReLU(0.2, True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),  #
        #
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
        #     norm_layer(64),
        #     nn.LeakyReLU(0.2, True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
        #     norm_layer(64),
        #     nn.LeakyReLU(0.2, True),
        #
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
        #     norm_layer(64),
        #     nn.LeakyReLU(0.2, True),
        # )
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, padding=0, stride=3),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64, momentum=1, affine=True),
            nn.ReLU())
        self.classifier = MML_Metric(num_classes=num_classes,neighbor_k=neighbor_k, batch_size=batch_size, shot_num=shot_num,query_num = query_num)  # 1*num_classes

    def forward(self, input1, input2):


        # # extract features of input1--query image
        # q = self.features(input1).contiguous()
        # q = q.view(q.size(0), q.size(1), -1)
        # q = q.permute(0, 2, 1)
        #
        # # extract features of input2--support set
        # S = self.features(input2).contiguous()
        # S = S.view(S.size(0), S.size(1), -1)
        # S = S.permute(0, 2, 1)
        # extract features of input1--query image
        # q = self.features(input1).contiguous()#input1(300,3,84,84) input2(20,3,84,84)
        out = self.layer1(input1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # (75,64,83)
        q = out.view(out.size(0), out.size(1), -1)  # (300,64,21,21)->(300,64,441)
        q = q.permute(0, 2, 1)  # (75,83,64)

        # extract features of input2--support set
        # S = self.features(input2).contiguous()#(20,64,21,21)
        out2 = self.layer1(input2)
        out2 = self.layer2(out2)
        out2 = self.layer3(out2)
        out2 = self.layer4(out2)  # (5,64,83)
        S = out2.view(out2.size(0), out2.size(1), -1)  # (20,64,441)
        S = S.permute(0, 2, 1)  # (5,83,64)

        #x, Q_S_List = self.classifier(q, S)
        x = self.classifier(q, S)

        #return x, Q_S_List
        return x


# ========================== Define MML ==========================#
class MML_Metric(nn.Module):
    def __init__(self, num_classes=5, neighbor_k=3, batch_size=4, shot_num=1,query_num=15):
        super(MML_Metric, self).__init__()
        self.neighbor_k = neighbor_k
        self.batch_size = batch_size
        self.shot_num = shot_num
        self.query_num = query_num
        self.num_class = num_classes

        self.Norm_layer = nn.BatchNorm1d(num_classes * 3, affine=True)
        self.FC_layer = nn.Conv1d(1, 1, kernel_size=3, stride=1, dilation=5, bias=False)

    def cal_covariance_matrix_Batch(self, feature):
        n_local_descriptor = torch.tensor(feature.size(1)).cuda()
        feature_mean = torch.mean(feature, 1, True)
        feature = feature - feature_mean
        cov_matrix = torch.matmul(feature.permute(0, 2, 1), feature)
        cov_matrix = torch.div(cov_matrix, n_local_descriptor - 1)
        cov_matrix = cov_matrix + 0.01 * torch.eye(cov_matrix.size(1)).cuda()

        return feature_mean, cov_matrix

    def support_remaining(self, S):

        S_new = []
        for ii in range(S.size(0)):
            indices = [j for j in range(S.size(0))]
            indices.pop(ii)
            indices = torch.tensor(indices).cuda()

            S_clone = S.clone()
            S_remain = torch.index_select(S_clone, 0, indices.long())
            S_remain = S_remain.contiguous().view(-1, S_remain.size(2))
            S_new.append(S_remain.unsqueeze(0))

        S_new = torch.cat(S_new, 0)
        return S_new

    def KL_distance_Batch(self, mean1, cov1, mean2, cov2):

        cov2_inverse = torch.inverse(cov2)
        mean_diff = mean1 - mean2.squeeze(1)
        # Calculate the trace
        matrix_product = torch.matmul(cov1.unsqueeze(1), cov2_inverse)
        matrix_product = matrix_product.contiguous().view(self.query_num*self.num_class, self.num_class, 64, 64)
        trace_dis = [torch.trace(matrix_product[j][i]).unsqueeze(0) for j in range(matrix_product.size(0)) for i in
                     range(matrix_product.size(1))]
        trace_dis = torch.cat(trace_dis, 0)
        trace_dis = trace_dis.view(matrix_product.size(0), matrix_product.size(1))

        # Calcualte the Mahalanobis Distance
        maha_product = torch.matmul(mean_diff.unsqueeze(2), cov2_inverse)
        maha_product = torch.matmul(maha_product, mean_diff.unsqueeze(3))
        maha_product = maha_product.squeeze(3)
        maha_product = maha_product.squeeze(2)
        maha_product = maha_product.contiguous().view(-1, 5)
        # matrix_det = torch.logdet(cov2) - torch.logdet(cov1).unsqueeze(1)
        matrix_det = -(torch.slogdet(cov1).logabsdet - torch.slogdet(cov2).logabsdet.unsqueeze(1))

        KL_dis = trace_dis + maha_product + matrix_det - mean1.size(2)

        return KL_dis / 2.

    def cal_MML_similarity(self, input1_batch, input2_batch):

        Similarity_list = []
        Q_S_List = []
        input1_batch = input1_batch.contiguous().view(1, -1, input1_batch.size(1),
                                                      input1_batch.size(2))
        input2_batch = input2_batch.contiguous().view(1, -1, input2_batch.size(1),
                                                      input2_batch.size(2))

        for i in range(1):
            input1 = input1_batch[i]
            input2 = input2_batch[i]

            # L2 Normalization
            input1_norm = torch.norm(input1, 2, 2, True)
            input2_norm = torch.norm(input2, 2, 2, True)

            # Calculate the mean and covariance of the all the query images
            query_mean, query_cov = self.cal_covariance_matrix_Batch(
                input1)

            # Calculate the mean and covariance of the support set
            support_set = input2.contiguous().view(-1,
                                                   self.shot_num * input2.size(1), input2.size(2))
            s_mean, s_cov = self.cal_covariance_matrix_Batch(support_set)

            # Find the remaining support set
            support_set_remain = self.support_remaining(support_set)
            s_remain_mean, s_remain_cov = self.cal_covariance_matrix_Batch(
                support_set_remain)

            # Calculate the Wasserstein Distance
            kl_dis = -self.KL_distance_Batch(s_mean, s_cov,query_mean, query_cov)

            # Calculate the Image-to-Class Similarity
            query_norm = input1 / input1_norm
            support_norm = input2 / input2_norm
            assert (torch.min(input1_norm) > 0)
            assert (torch.min(input2_norm) > 0)

            support_norm_p = support_norm.permute(0, 2, 1)
            support_norm_p = support_norm_p.contiguous().view(-1,
                                                              self.shot_num * support_norm.size(2),
                                                              support_norm.size(1))

            support_norm_l = support_norm.contiguous().view(-1,
                                                            self.shot_num * support_norm.size(1),
                                                            support_norm.size(2))

            # local level and part level cosine similarity between a query set and a support set
            innerproduct_matrix_l = torch.matmul(query_norm.unsqueeze(1),
                                                 support_norm_l.permute(0, 2, 1))
            innerproduct_matrix_p = torch.matmul(query_norm.permute(0, 2, 1).unsqueeze(1),
                                                 support_norm_p.permute(0, 2, 1))

            # choose the top-k nearest neighbors
            topk_value_l, topk_index_l = torch.topk(innerproduct_matrix_l, self.neighbor_k, 3)
            inner_sim_l = torch.sum(torch.sum(topk_value_l, 3), 2)

            topk_value_p, topk_index_p = torch.topk(innerproduct_matrix_p, self.neighbor_k, 3)
            inner_sim_p = torch.sum(torch.sum(topk_value_p, 3), 2)


            # Using Fusion Layer to fuse three parts
            kl_sim_soft2 = torch.cat((kl_dis, inner_sim_l, inner_sim_p), 1)
            kl_sim_soft2 = self.Norm_layer(kl_sim_soft2).unsqueeze(1)
            kl_sim_soft = self.FC_layer(kl_sim_soft2).squeeze(1)

            Similarity_list.append(kl_sim_soft)

        #return Similarity_list, Q_S_List
        return Similarity_list

    def forward(self, x1, x2):

        #Similarity_list, Q_S_List = self.cal_MML_similarity(x1, x2)#x1 x2(75,83,64) (15,83,64)
        Similarity_list= self.cal_MML_similarity(x1, x2)

        #return Similarity_list, Q_S_List
        return Similarity_list

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


