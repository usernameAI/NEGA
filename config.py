# -*- coding: utf-8 -*-
import argparse


def config():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--embed_dim', type=int, default=8, metavar='N', help='embedding size')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N', help='dropout probability')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=9999, metavar='N', help='number of epochs to train')
    parser.add_argument('--uu_gnn_layer', type=int, default=1, metavar='N', help='number of layers for UI GNN')
    parser.add_argument('--ui_gnn_layer', type=int, default=1, metavar='N', help='number of layers for UI GNN')
    parser.add_argument('--uu_hidden_space', type=int, default=1, metavar='N', help='number of ui_hidden_space')
    parser.add_argument('--ui_hidden_space', type=int, default=1, metavar='N', help='number of ui_hidden_space')
    parser.add_argument('--mlp_layer', type=int, default=1, metavar='N', help='number of layers for final UI MLP')
    parser.add_argument('--dataset', type=str, default='filmtrust', metavar='DATA', help='name of dataset')
    args = parser.parse_args()

    return args
