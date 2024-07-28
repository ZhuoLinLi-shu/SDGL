import torch
import numpy as np
import argparse
import time
from SDGL.Pems4.util import *
import matplotlib.pyplot as plt
from SDGL.Pems4.engine import trainer
from torch import nn
from SDGL.Pems4.lib.dataloader import get_dataloader
from SDGL.Pems4.lib.metrics import All_Metrics

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--dataset', type=str, default='PEMSD4', help='dataset')

parser.add_argument('--gcn_bool', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--addaptadj', type=bool, default=True, help='whether add adaptive adj')

parser.add_argument('--seq_length', type=int, default=12, help='input time windows length')
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=307, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')  # 0.001
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0., help='weight decay rate')  # 0.0001
parser.add_argument('--epochs', type=int, default=200, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./model/pems4', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--log_file', type=str, default='./log/pems4_log', help='log file')
parser.add_argument('--embed_dim', type=int, default=10, help='node dim')  # 40
parser.add_argument('--rate', type=int, default=1, help='')  # 10

parser.add_argument('--dropout_ingc', type=float, default=0.3, help='Dropout in Dynamic Graph Learning Module')

parser.add_argument('--eta', type=float, default=1, help='useless, ignore it')
parser.add_argument('--gamma', type=float, default=0.0001, help='γ in Eq.2')
parser.add_argument('--order', type=float, default=1, help='the Weight of the graph loss')
parser.add_argument('--moco', type=float, default=0.99, help='hyperpa-parameter m in momentum update')
parser.add_argument('--layers', type=int, default=3, help='number of layers')

parser.add_argument('--column_wise', type=bool, default=False)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--val_ratio', type=float, default=0.2)
parser.add_argument('--lag', type=int, default=12, help='input time windows length')
parser.add_argument('--horizon', type=int, default=12, help='predict window length')
parser.add_argument('--dilation_exponential_', type=int, default=1)

args = parser.parse_args()
log = open(args.log_file, 'w') # + '-pems4'  + 'no_moco'
torch.set_num_threads(3)


def log_string(string, log=log):
    log.write(string + '\n')
    log.flush()
    print(string)


def main():
    # set seed
    args.seed = int(time.time())
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    # load data
    device = torch.device(args.device)

    train_dataloader, val_dataloader, test_dataloader, scaler, test_realy = get_dataloader(args, 'std', single=False)
    print('loda dataset done')

    log_string(str(args))

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, args.gcn_bool, args.addaptadj,
                     args.embed_dim, args.dropout_ingc, args.eta, args.gamma, args.order, args.moco,
                     args.layers, args.batch_size, args.dilation_exponential_)
    nparams = sum([p.nelement() for p in engine.model.parameters()])
    log_string('The model parameter number is: {}'.format(str(nparams)))

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()

        for iter, (x, y) in enumerate(train_dataloader):
            # x = B, T, N, in_dim,
            trainx = x[..., :1]
            trainx = trainx.transpose(1, 3)
            trainy = y[..., :1]
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy, pred_time_embed=None, iter=iter)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, {:.4f} :Train Loss, {:.4f}:Train MAPE, {:.4f}: Train RMSE'
                log_string(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))

        t2 = time.time()
        train_time.append(t2 - t1)
        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(val_dataloader):
            trainx = x[..., :1]
            trainx = trainx.transpose(1, 3)
            trainy = y[..., :1]
            trainy = trainy.transpose(1, 3)
            metrics = engine.eval(trainx, trainy, pred_time_embed=None)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        engine.scheduler.step(np.mean(valid_loss))

        s2 = time.time()
        # log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        # print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, ' \
              'Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        log_string(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse,
                              mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)))

        torch.save(engine.model.state_dict(),
                   args.save + "_epoch_" + str(i) + ".pth")

    log_string("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    realy = []

    for _, (_, y) in enumerate(test_dataloader):
        realy.append(y[..., :1].transpose(1, 3))
    realy = torch.cat(realy, dim=0).squeeze(1)

    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "_epoch_" + str(bestid + 1) + ".pth"))
    engine.model.eval()
    outputs = []
    for iter, (x, y) in enumerate(test_dataloader):
        testx = x[..., :1]
        testx = testx.transpose(1, 3)
        testx = nn.functional.pad(testx, (1, 0, 0, 0))

        with torch.no_grad():
            preds, _, _ = engine.model(testx, pred_time_embed=None)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    log_string("The valid loss on best model is {}".format(str(round(his_loss[bestid], 4))))

    amae = []
    amape = []
    armse = []
    # print('seed is {}'.format(args.seed))
    log_string('seed is {}'.format(args.seed))
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = scaler.inverse_transform(realy[:, :, i])
        # metrics = metric(pred, real)
        metrics = All_Metrics(pred, real, None, 0.)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'

        log_string(log.format(i + 1, metrics[0], metrics[2], metrics[1]))
        amae.append(metrics[0].item())
        amape.append(metrics[2].item())
        armse.append(metrics[1].item())

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    log_string(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))
    torch.save(engine.model.state_dict(),
               args.save + "_exp" + str(args.expid) + "_best_" + str(args.order) + '_' + str(args.seed)+ ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
