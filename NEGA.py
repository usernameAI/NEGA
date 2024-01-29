import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect
from torch.nn.init import kaiming_normal_
import config
import numpy as np

args = config.config()


class NEGA(nn.Module):
    def __init__(self, num_users, num_items, num_ratings, history_u, history_i, history_ur, history_ir, embed_dim, social_neighbor, cuda='cuda'):
        super(NEGA, self).__init__()
        self.embed_dim = embed_dim
        u2e = nn.Embedding(num_users, self.embed_dim)
        i2e = nn.Embedding(num_items, self.embed_dim)
        r2e = nn.Embedding(num_ratings, self.embed_dim)
        self.enc_u = UI_Aggregator(i2e, r2e, u2e, embed_dim, history_u, history_ur, cuda, user=True)
        self.enc_i = UI_Aggregator(i2e, r2e, u2e, embed_dim, history_i, history_ir, cuda, user=False)
        self.enc_social = Social_Aggregator(None, u2e, embed_dim, social_neighbor, cuda)
        self.user_agg = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim)
        self.user_agg_list = nn.ModuleList(MLP(self.embed_dim, self.embed_dim, self.embed_dim) for _ in range(args.mlp_layer - 1))
        self.item_agg_list = nn.ModuleList(MLP(self.embed_dim, self.embed_dim, self.embed_dim) for _ in range(args.mlp_layer))
        self.criterion = nn.MSELoss()
        self.predict_linear = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.predict_act = nn.Sigmoid()
        self.predict_bn = nn.BatchNorm1d(self.embed_dim)
        self.final_linear = nn.Linear(self.embed_dim, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and module.bias is None:
            kaiming_normal_(module.weight.data, nonlinearity='selu')
        if isinstance(module, nn.Linear) and module.bias is not None:
            kaiming_normal_(module.weight.data, nonlinearity='selu')
            module.bias.data.zero_()

    def forward(self, nodes_u, nodes_i):
        item_space = self.enc_u(nodes_u)
        social_space = self.enc_social(nodes_u)
        user_latent_feature0 = torch.cat((item_space, social_space), dim=-1)
        user_latent_feature = self.user_agg(user_latent_feature0) + item_space + social_space
        if args.mlp_layer > 1:
            for layer in range(args.mlp_layer - 1):
                user_latent_feature = self.user_agg_list[layer](user_latent_feature)
        item_latent_feature = self.enc_i(nodes_i)
        for layer in range(args.mlp_layer):
            item_latent_feature = self.item_agg_list[layer](item_latent_feature)
        latent_feature = torch.cat((user_latent_feature, item_latent_feature, user_latent_feature * item_latent_feature), -1)
        epsilon = self.predict_act(self.predict_linear(latent_feature))
        latent_feature = epsilon * user_latent_feature + (1 - epsilon) * item_latent_feature
        latent_feature = self.predict_bn(latent_feature)
        scores = self.final_linear(latent_feature)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_i, ratings):
        scores = self.forward(nodes_u, nodes_i)
        return self.criterion(scores, ratings)


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim // 4)
        self.att3 = nn.Linear(self.embed_dim // 4, 1)
        self.linear1 = nn.Linear(self.embed_dim, 1)

    def forward(self, node1, u_rep, num_neighs):
        uv_reps = u_rep.repeat(num_neighs, 1)
        x = torch.cat((node1, uv_reps), -1)
        weight = self.linear1(node1)
        weight = torch.sigmoid(weight) + 1
        x = F.selu(self.att1(x))
        x = F.selu(self.att2(x))
        x = self.att3(x)
        att = entmax_bisect(x, alpha=weight)
        return att


class UI_Aggregator(nn.Module):
    def __init__(self, i2e, r2e, u2e, embed_dim, history_ui, history_r, cuda="cuda", user=True):
        super(UI_Aggregator, self).__init__()
        self.user = user
        self.i2e = i2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.history_ui = history_ui
        self.history_r = history_r
        self.attention_list = nn.ModuleList(
            Attention(self.embed_dim) for _ in range(args.ui_hidden_space)
        )
        self.gate_linear = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.gate_act = nn.Sigmoid()
        self.gate_linear1 = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.gate_act1 = nn.Sigmoid()
        self.GAT_MLP = MLP(self.embed_dim, self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        ui_history = []
        r_history = []
        for node in nodes:
            ui_history.append(self.history_ui[int(node)])
            r_history.append(self.history_r[int(node)])
        num_len = len(ui_history)
        embed_matrix = torch.empty(num_len, self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(num_len):
            history = ui_history[i]
            num_histroy_ui = len(history)
            tmp_label = r_history[i]
            if self.user == True:
                e_ui = self.i2e.weight[history]
                ui_rep = self.u2e.weight[nodes[i]]
            else:
                e_ui = self.u2e.weight[history]
                ui_rep = self.i2e.weight[nodes[i]]
            e_r = self.r2e.weight[tmp_label]
            x = torch.cat((e_ui, e_r, e_ui * e_r), -1)
            alpha = self.gate_act(self.gate_linear(x))
            o_history = alpha * e_ui + (1 - alpha) * e_r
            attention_feature_list = []
            for hidden_space in range(args.ui_hidden_space):
                o_history = F.normalize(o_history, dim=-1)
                ui_rep = F.normalize(ui_rep, dim=-1)
                att_weight = self.attention_list[hidden_space](o_history, ui_rep, num_histroy_ui)
                ui_rep = torch.mm(o_history.t(), att_weight)
                ui_rep = ui_rep.t()
                attention_feature_list.append(ui_rep)
            att_history = sum(attention_feature_list) / len(attention_feature_list)
            embed_matrix[i] = att_history
        neigh_feats = embed_matrix
        neigh_feats = self.GAT_MLP(neigh_feats)
        if self.user == True:
            self_feats = self.u2e.weight[nodes]
        else:
            self_feats = self.i2e.weight[nodes]
        combined = torch.cat((self_feats, neigh_feats, self_feats * neigh_feats), dim=-1)
        beta = self.gate_act1(self.gate_linear1(combined))
        combined_feats = beta * self_feats + (1 - beta) * neigh_feats
        return combined_feats


class Social_Aggregator(nn.Module):
    def __init__(self, features, u2e, embed_dim, social_neighbor, cuda="cuda"):
        super(Social_Aggregator, self).__init__()
        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.social_neighbor = social_neighbor
        self.attention_list = nn.ModuleList(
            Attention(self.embed_dim) for _ in range(args.uu_hidden_space)
        )
        self.gate_linear = nn.Linear(3 * self.embed_dim, self.embed_dim)
        self.gate_act = nn.Sigmoid()
        self.GAT_MLP = MLP(self.embed_dim, self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        combined_feats_list = []
        for layer in range(args.uu_gnn_layer):
            to_neighs = []
            count = layer
            for node in nodes:
                final_neigh = self.social_neighbor[int(node)]
                while count > 0:
                    other_neigh = []
                    for other_node in final_neigh:
                        other_neigh.extend(self.social_neighbor[int(other_node)])
                    final_neigh = final_neigh + other_neigh
                    '''列表去重'''
                    final_neigh = np.array(final_neigh, dtype=object)
                    final_neigh = np.unique(final_neigh)
                    final_neigh = final_neigh.tolist()
                    count = count - 1
                to_neighs.append(final_neigh)
            num_len = len(nodes)
            embed_matrix = torch.empty(num_len, self.embed_dim, dtype=torch.float).to(self.device)
            for i in range(num_len):
                tmp_adj = to_neighs[i]
                if isinstance(tmp_adj, int):
                    num_neighs = list(num_neighs)
                else:
                    num_neighs = len(tmp_adj)
                e_u = self.u2e.weight[list(tmp_adj)]
                u_rep = self.u2e.weight[nodes[i]]
                attention_feature_list = []
                for hidden_space in range(args.ui_hidden_space):
                    e_u = F.normalize(e_u, dim=-1)
                    u_rep = F.normalize(u_rep, dim=-1)
                    att_weight = self.attention_list[hidden_space](e_u, u_rep, num_neighs)
                    u_rep = torch.mm(e_u.t(), att_weight)
                    u_rep = u_rep.t()
                    attention_feature_list.append(u_rep)
                att_history = sum(attention_feature_list) / len(attention_feature_list)
                embed_matrix[i] = att_history
            neigh_feats = embed_matrix
            neigh_feats = self.GAT_MLP(neigh_feats)
            self_feats = self.u2e.weight[nodes]
            combined = torch.cat((self_feats, neigh_feats, self_feats * neigh_feats), dim=-1)
            gama = self.gate_act(self.gate_linear(combined))
            combined_feats = gama * self_feats + (1 - gama) * neigh_feats
            combined_feats_list.append(combined_feats)
        combined_feats = sum(combined_feats_list) / len(combined_feats_list)
        return combined_feats


class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super(MLP, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.input_projection = nn.Linear(input_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, output_dim)
        self.activation = nn.SELU()
        self.dropout = nn.Dropout(p=args.dropout)
        self.bn1 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.input_projection(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.output_projection(x)
        return x
