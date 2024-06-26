from __future__ import division
from __future__ import print_function
import os
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from model import PLCSR, Discriminator
from optimizer import loss_function,dis_loss
from utils import mask_gae_edges, preprocess_graph, weights_init, load_ds, load_data
from sklearn.cluster import KMeans
from metric import cluster_accuracy
from optim_weight import WeightEMA
import random
import copy
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='cora', help='Type of dataset.')
parser.add_argument('--sds', type=int, default=0, help='Whether to use small datasets, 0/1.')
parser.add_argument('--lr_dec', type=float, default=0.15, help='Initial learning rate.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--lr_dis', type=float, default=0.001, help='Discriminator learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--gpu', type=int, default=0, help='GPU id.')
parser.add_argument('--model', type=str, default='arvga', help="Models used")
parser.add_argument('--arga', type=int, default=1, help='Whether to use arga, 0/1.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--hidden3', type=int, default=64, help='Number of units in hidden layer 3.')
parser.add_argument('--save_ckpt', type=int, default=0, help='Whether to save checkpoint, 0/1.')
parser.add_argument('--use_ckpt', type=int, default=0, help='Whether to use checkpoint, 0/1.')
parser.add_argument('--optimi', type=str, default='SGD', help="Optimizers used, [SGD ADAM]")
parser.add_argument('--e', type=int, default=152, help='Number of epochs to pretrain.')
parser.add_argument('--e1', type=int, default=250, help='Number of epochs.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')

parser.add_argument('--alpha', type=float, default=0.999, help='Rate of updating teacher parameters.')
parser.add_argument('--w_re', type=float, default=0.01, help='Weight of lossre.')
parser.add_argument('--w_pq', type=float, default=0, help='Weight of losspq.')
parser.add_argument('--w_con', type=float, default=2.0, help='Weight of losscon.')
parser.add_argument('--w_pos', type=float, default=1, help='Weight of losspos.')
parser.add_argument('--w_nega', type=float, default=0.005, help='Weight of lossnega.')

parser.add_argument('--edp', type=float, default=0, help='Edge drop percent.')
parser.add_argument('--fdp', type=float, default=0, help='Feature drop percent.')
parser.add_argument('--sta_high', type=float, default=0.95,help='Static high threshold.')
parser.add_argument('--ini_dyn', type=float, default=0.96,help='Initial dynamic threshold.')
parser.add_argument('--tt', type=float, default=0.3,help='Dynamic teacher threshold.')
parser.add_argument('--st', type=float, default=0.2,help='Dynamic student threshold.')
parser.add_argument('--sta_low', type=float, default=0.2,help='Static low threshold.')
parser.add_argument('--replace', type=int, default=1, help='Sampling with replacement, 0/1.')
parser.add_argument('--encoder', type=str, default='a', help="Encoder for selecting nodes, [s, t, c]")

args = parser.parse_args()

def main():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # load data
    if args.sds == 1:
        adj, features, labels = load_ds(args.ds)
    else:
        adj, features, labels = load_data(args.ds)

    #data augmentation
    adj_S = adj
    if args.edp != 0:
        adj_S = aug_random_edge(adj, drop_percent=args.edp)
    features_S = features
    if args.fdp != 0:
        features_S = aug_random_fea(features, drop_percent=args.fdp)

    # Some preprocessing
    adj_T = preprocess_graph(mask_gae_edges(adj))
    adj_S = preprocess_graph(mask_gae_edges(adj_S))
    cluster_num = labels.max().item() + 1
    n_nodes, feat_dim = features.shape
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    # build model
    if args.model == 'arvga':
        model_S = PLCSR(feat_dim, args.hidden1, args.hidden2, args.dropout)
        model_T = PLCSR(feat_dim, args.hidden1, args.hidden2, 0)
    modeldis_S = Discriminator(args.hidden1, args.hidden2, args.hidden3)
    modeldis_S.apply(weights_init)
    modeldis_T = Discriminator(args.hidden1, args.hidden2, args.hidden3)
    modeldis_T.apply(weights_init)
    student_params = list(model_S.parameters())
    teacher_params = list(model_T.parameters())
    for param in teacher_params:
        param.requires_grad = False

    # build optimizer
    optimizer_S = optim.Adam(student_params, lr=args.lr)
    optimizer_dis_S = optim.Adam(modeldis_S.parameters(), lr=args.lr_dis, betas=(0.5, 0.9))
    optimizer_dis_T = optim.Adam(modeldis_T.parameters(), lr=args.lr_dis, betas=(0.5, 0.9))

    cluster_centers = None
    cluster_centers2 = None
    optimizer_dec_S = None
    pseudo_labels = torch.full(labels.shape, -1, dtype=torch.long)
    n_pseudo_labels = torch.ones([n_nodes, cluster_num])
    args.st = args.tt - 0.1
    args.ini_dyn = args.sta_high - 0.1
    idx_con = []
    loss_pos = loss_nega = 0
    saccs=[]
    accs=[]
    nums=[]

    use_gpu = 1
    if args.ds == 'pubmed':
        use_gpu = 0
    if use_gpu == 1:
        pos_weight = pos_weight.cuda()
        features = features.cuda()
        features_S = features_S.cuda()
        adj_T = adj_T.cuda()
        adj_S = adj_S.cuda()
        adj_label = adj_label.cuda()
        model_S = model_S.cuda()
        modeldis_S = modeldis_S.cuda()
        model_T = model_T.cuda()
        modeldis_T = modeldis_T.cuda()
        pseudo_labels = pseudo_labels.cuda()
        n_pseudo_labels = n_pseudo_labels.cuda()

    ee=0
    if args.use_ckpt == 1:
        ee = args.e1
        checkpoint = torch.load('checkpoint_{}.pkl'.format(args.ds))
        model_S.load_state_dict(checkpoint['models_save'])
        modeldis_S.load_state_dict(checkpoint['modelds_save'])
        model_S.eval()
        recovered, mu, logvar, z = model_S(features, adj_T)
        hidden_emb = mu.cpu().data
        kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(hidden_emb)
        if use_gpu == 1:
            cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)).cuda(),
                                       requires_grad=True)
            cluster_centers2 = torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor).cuda()
        if use_gpu == 0:
            cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)),
                                       requires_grad=True)
            cluster_centers2 = torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)
        if args.optimi == 'SGD':
            optimizer_dec_S = torch.optim.SGD(list(model_S.parameters()) + [cluster_centers], lr=args.lr_dec)
        else:
            optimizer_dec_S = torch.optim.Adam(list(model_S.parameters()) + [cluster_centers], lr=args.lr_dec)


    for epoch in range(ee, args.epochs+1):
        print('-' * 30)
        print(epoch)
        model_S.train()
        model_T.train()
        # pretraining
        if epoch < args.e:
            optimizer_S.zero_grad()
            recovered, mu, logvar, z = model_S(features, adj_T)
            hidden_emb = mu
            if args.arga == 1:
                modeldis_S.train()
                for j in range(10):
                    z_real_dist = np.random.randn(adj.shape[0], args.hidden2)
                    z_real_dist = torch.FloatTensor(z_real_dist)
                    if use_gpu == 1:
                        z_real_dist = z_real_dist.cuda()
                    d_real = modeldis_S(z_real_dist)
                    d_fake = modeldis_S(hidden_emb)
                    optimizer_dis_S.zero_grad()
                    dis_loss_ = dis_loss(d_real, d_fake)
                    dis_loss_.backward(retain_graph=True)
                    optimizer_dis_S.step()
            loss = loss_function(preds=model_S.dc(z), labels=adj_label, n_nodes=n_nodes, norm=norm,
                                 pos_weight=pos_weight, mu=mu, logvar=logvar, )
            loss.backward()
            optimizer_S.step()
            hidden_emb = hidden_emb.cpu().data
            kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(hidden_emb)
            cluster_pred = kmeans.predict(hidden_emb)
            acc, nmi, f1 = cluster_accuracy(cluster_pred, labels.cpu(), cluster_num)
            print('Pretrained model result: {:.2f},{:.2f},{:.2f}'.format(acc*100, nmi*100, f1*100))


            model_S.eval()
            recovered, mu, logvar, z = model_S(features, adj_T)
            hidden_emb = mu.cpu().data
            kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=20).fit(hidden_emb)
            cluster_pred = kmeans.predict(hidden_emb)
            acc, nmi, f1 = cluster_accuracy(cluster_pred, labels.cpu(), cluster_num)
            print('Eval model result: {:.2f},{:.2f},{:.2f}'.format(acc * 100, nmi * 100, f1 * 100))

            if epoch == args.epoch and args.save_ckpt == 1:
                torch.save({'models_save': model_S.state_dict(),
                            'modelds_save': modeldis_S.state_dict(),
                            },
                           'checkpoint_{}.pkl'.format(args.ds))

            if epoch == args.e-1:
                if use_gpu == 1:
                    cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor))
                                               .cuda(), requires_grad=True)
                    cluster_centers2 = Variable(
                        (torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)).cuda(),
                        requires_grad=False)
                if use_gpu == 0:
                    cluster_centers = Variable((torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)),
                                               requires_grad=True)
                    cluster_centers2 = Variable(
                        (torch.from_numpy(kmeans.cluster_centers_).type(torch.FloatTensor)),
                        requires_grad=False)

                if args.optimi == 'SGD':
                    optimizer_dec_S = torch.optim.SGD(list(model_S.parameters()) + [cluster_centers], lr=args.lr_dec)
                else:
                    optimizer_dec_S = torch.optim.Adam(list(model_S.parameters()) + [cluster_centers], lr=args.lr_dec)

        #train
        else:
            recovered_SS, mu_SS, logvar_SS, z_SS = model_S(features_S, adj_S)
            recovered_TT, mu_TT, logvar_TT, z_TT = model_T(features, adj_T)
            recovered, mu, logvar, z = model_S(features, adj_T)

            hidden_emb_SS = mu_SS
            if args.arga == 1:
                modeldis_S.train()
                for j in range(10):
                    z_real_dist = np.random.randn(adj.shape[0], args.hidden2)
                    z_real_dist = torch.FloatTensor(z_real_dist)
                    if use_gpu == 1:
                        z_real_dist = z_real_dist.cuda()
                    d_real = modeldis_S(z_real_dist)
                    d_fake = modeldis_S(hidden_emb_SS)
                    optimizer_dis_S.zero_grad()
                    dis_loss_ = dis_loss(d_real, d_fake)
                    dis_loss_.backward(retain_graph=True)
                    optimizer_dis_S.step()

            hidden_emb_TT = mu_TT
            if args.arga == 1:
                modeldis_T.train()
                for j in range(10):
                    z_real_dist = np.random.randn(adj.shape[0], args.hidden2)
                    z_real_dist = torch.FloatTensor(z_real_dist)
                    if use_gpu == 1:
                        z_real_dist = z_real_dist.cuda()
                    d_real = modeldis_T(z_real_dist)
                    d_fake = modeldis_T(hidden_emb_TT)
                    optimizer_dis_T.zero_grad()
                    dis_loss_ = dis_loss(d_real, d_fake)
                    dis_loss_.backward(retain_graph=True)
                    optimizer_dis_T.step()

            hidden_emb = mu
            if args.arga == 1:
                modeldis_S.train()
                for j in range(10):
                    z_real_dist = np.random.randn(adj.shape[0], args.hidden2)
                    z_real_dist = torch.FloatTensor(z_real_dist)
                    if use_gpu == 1:
                        z_real_dist = z_real_dist.cuda()
                    d_real = modeldis_S(z_real_dist)
                    d_fake = modeldis_S(hidden_emb)
                    optimizer_dis_S.zero_grad()
                    dis_loss_ = dis_loss(d_real, d_fake)
                    dis_loss_.backward(retain_graph=True)
                    optimizer_dis_S.step()

            optimizer_dec_T = WeightEMA(teacher_params + [cluster_centers2], student_params + [cluster_centers],
                                        alpha=args.alpha * (epoch / args.epochs))
            optimizer_dec_S.zero_grad()
            loss_re = loss_function(preds=model_S.dc(z), labels=adj_label, n_nodes=n_nodes, norm=norm,
                                   pos_weight=pos_weight, mu=mu, logvar=logvar)
            loss = args.w_re * loss_re

            losspq, p, q = loss_func(mu, cluster_centers)
            cluster_pred_score, cluster_pred = dist_2_label(q)
            losspq_SS, p_SS, q_SS = loss_func(mu_SS, cluster_centers)
            cluster_pred_score_SS, cluster_pred_SS = dist_2_label(q_SS)
            losspq_TT, p_TT, q_TT = loss_func(mu_TT, cluster_centers2)
            cluster_pred_score_TT, cluster_pred_TT = dist_2_label(q_TT)
            if epoch > args.e1:
                loss += args.w_pq * losspq
            acc, nmi, f1 = cluster_accuracy(cluster_pred.cpu(), labels.cpu(), cluster_num)
            print('Trained model result: {:.2f},{:.2f},{:.2f}'.format(acc*100, nmi*100, f1*100))

            if epoch == args.e1 + 1:
                cluster_pred = torch.tensor(cluster_pred)
                if use_gpu == 1:
                    cluster_pred = cluster_pred.cuda()
                idx_sta = []
                for i in range(n_nodes):
                    if cluster_pred_score[i] > args.sta_high:
                        idx_sta.append(i)
                        pseudo_labels[i] = cluster_pred[i]
                sacc_, _, _ = cluster_accuracy(pseudo_labels[idx_sta].cpu(), labels[idx_sta].cpu(), cluster_num)
                idx_con_orig = idx_sta

            if epoch > args.e1:
                idx_pos = idx_con_orig + idx_con
                sacc, _, _ = cluster_accuracy(pseudo_labels[idx_pos].cpu(), labels[idx_pos].cpu(), cluster_num)
                if args.w_pos != 0 and epoch % 10 == 0:
                    if args.replace == 1:
                        idx_con = []
                    for i in range(n_nodes):
                        if i not in idx_con_orig and i not in idx_con:
                            if args.encoder == 's':
                                if cluster_pred_score_SS[i] > args.ini_dyn - args.st * epoch / args.epochs:
                                    idx_con.append(i)
                                    pseudo_labels[i] = cluster_pred[i]
                            if args.encoder == 't':
                                if cluster_pred_score_TT[i] > args.ini_dyn - args.tt * epoch / args.epochs:
                                    idx_con.append(i)
                                    pseudo_labels[i] = cluster_pred[i]
                            if args.encoder == 'c':
                                if (cluster_pred_score_TT[i] > args.ini_dyn - args.tt * epoch / args.epochs) and \
                                        (cluster_pred_score_SS[i] > args.ini_dyn - args.st * epoch / args.epochs) and \
                                        (cluster_pred_TT[i] == cluster_pred_SS[i]):
                                    idx_con.append(i)
                                    pseudo_labels[i] = cluster_pred[i]
                    nums.append(len(idx_pos))
                    saccs.append(float('{:.4f}'.format(sacc)))
                if args.w_pos != 0:
                    loss_pos = F.nll_loss(torch.log(q[idx_pos]), pseudo_labels[idx_pos])
                    loss_con = F.mse_loss(q_SS, q_TT, reduction='mean')
                    loss_pos = loss_pos + loss_con * (epoch / args.epochs) * args.w_con
                else:
                    loss_pos = 0

                if args.w_nega != 0:
                    idx_nega = set()
                    for i in range(n_nodes):
                        if i not in idx_pos and cluster_pred_score[i] < args.sta_low:
                            idx_nega.add(i)
                    idx_nega = list(idx_nega)
                    if len(idx_nega) == 0:
                        loss_nega = 0
                    else:
                        loss_nega = torch.mean(-torch.sum(torch.mul(torch.log(1 - q)[idx_nega],
                                                                    n_pseudo_labels[idx_nega]), dim=1))
            loss += args.w_pos * loss_pos + args.w_nega * loss_nega
            print('loss re pq pos nega', loss_re, losspq, loss_pos, loss_nega)
            loss.backward()
            optimizer_dec_S.step()
            optimizer_dec_T.step()


            # test
            model_S.eval()
            model_T.eval()
            recovered, mu, logvar, z = model_S(features, adj_T)
            loss, p,q = loss_func(mu, cluster_centers)
            cluster_pred_score, cluster_pred = dist_2_label(q)
            acc, nmi, f1 = cluster_accuracy(cluster_pred.cpu(), labels.cpu(), cluster_num)
            accs.append(float('{:.4f}'.format(acc)))
            print('Eval model result:{:.2f},{:.2f},{:.2f}'.format(acc*100, nmi*100, f1*100))
    print(args)
    print('Cluster accuracy:', accs)
    print('Pseudo-label accuracy:', saccs)
    print('Number of pseudo-labels:', nums)



def aug_random_edge(input_adj, drop_percent=0.2):
    row_idx, col_idx = input_adj.nonzero()
    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    single_index_list = []
    for i in list(index_list):
        single_index_list.append(i)
        index_list.remove((i[1], i[0]))

    edge_num = int(len(row_idx) / 2)
    add_drop_num = int(edge_num * drop_percent / 2)
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0
    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj

def aug_random_fea(input_feature, drop_percent=0.05):
    node_num = input_feature.shape[0]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0])
    for j in mask_idx:
        aug_feature[j] = zeros
    return aug_feature

def loss_func(feat,cluster_centers):
    alpha = 1.0
    q = 1.0 / (1.0 + torch.sum((feat.unsqueeze(1) - cluster_centers) ** 2, dim=2) / alpha)
    q = q ** (alpha + 1.0) / 2.0
    q = (q.t() / torch.sum(q, dim=1)).t()

    weight = q ** 2 / torch.sum(q, dim=0)
    p = (weight.t() / torch.sum(weight, dim=1)).t()
    p = p.detach()

    log_q = torch.log(q)
    loss = F.kl_div(log_q, p)
    return loss, p, q


def dist_2_label(q_t):
    maxlabel, label = torch.max(q_t, dim=1)
    return maxlabel, label

def init():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    init()
    main()
