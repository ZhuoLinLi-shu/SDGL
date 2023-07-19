import torch
import numpy as np
import argparse
import time
import configparser
from SDGL.Time_series.engine import trainer
from torch import nn
from SDGL.Time_series.util_series import DataLoaderS, generate_metric

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--dataset', type=str, default='data/solar_AL.txt', help='dataset')

parser.add_argument('--gcn_bool', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--addaptadj', type=bool, default=True, help='whether add adaptive adj')

parser.add_argument('--seq_length', type=int, default=24*7, help='')
parser.add_argument('--nhid', type=int, default=16, help='') # 16
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=137, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')  # 0.001
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0000, help='weight decay rate')  # 0.0001
parser.add_argument('--epochs', type=int, default=50, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./model/time_series', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--log_file', type=str, default='./log/time_series', help='log file')
parser.add_argument('--embed_dim', type=int, default=10, help='node dim')
parser.add_argument('--rate', type=int, default=1, help='')

parser.add_argument('--dropout_ingc', type=float, default=0.3, help='Dropout in Dynamic Graph Learning Module')

parser.add_argument('--eta', type=float, default=1, help='useless, ignore it')
parser.add_argument('--gamma', type=float, default=0.001, help='γ in Eq.2')
parser.add_argument('--order', type=float, default=1, help='the Weight of the graph loss')
parser.add_argument('--moco', type=float, default=0.99, help='hyperpa-parameter m in momentum update')

parser.add_argument('--layers', type=int, default=3, help='number of layers')

parser.add_argument('--seq_out_len', type=int, default=1, help='predict window length')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--horizon', type=int, default=3)
parser.add_argument('--dilation_exponential', type=int, default=5)


args = parser.parse_args()
log = open(args.log_file, 'w')


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

    data = DataLoaderS(args.dataset, train=0.6, valid=0.2, device=device,
                       horizon=args.horizon, window=args.seq_length, normalize=args.normalize)
    print('loda dataset done')

    log_string(str(args))

    engine = trainer(data, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                     args.learning_rate, args.weight_decay, device, args.gcn_bool, args.addaptadj,
                     args.embed_dim, args.dropout_ingc, args.eta, args.gamma, args.order, args.moco,
                     args.seq_out_len, args.layers, args.batch_size, args.dilation_exponential)

    print("start training...", flush=True)
    log_string('paramter number is :' + str(sum([p.nelement() for p in engine.model.parameters()])))
    his_loss = []
    val_time = []
    train_time = []

    for i in range(1, args.epochs + 1):
        t1 = time.time()
        train_total_loss = 0
        n_samples = 0
        for iter, (x, y) in enumerate(data.get_batches(data.train[0], data.train[1], batch_size=args.batch_size)):
            # x = B, T, N, in_dim
            x = x.unsqueeze(-1)
            trainx = x.transpose(1, 3)
            trainy = y
            metrics = engine.train(trainx, trainy, data=data, pred_time_embed=None, iter=iter)
            train_total_loss += metrics
            n_samples += (y.size(0) * data.m)
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, {:.4f} :Train Loss'
                log_string(log.format(iter, metrics / (y.size(0) * data.m)))

        t2 = time.time()
        train_time.append(t2 - t1)

        # validation
        valid_total_loss = 0
        valid_total_loss_l1 = 0
        valid_n_samples = 0
        valid_output = None
        valid_label = None
        s1 = time.time()
        for iter, (x, y) in enumerate(data.get_batches(data.valid[0], data.valid[1], batch_size=args.batch_size)):
            trainx = x.unsqueeze(-1)
            trainx = trainx.transpose(1, 3)
            trainy = y

            metrics = engine.eval(trainx, trainy, data, pred_time_embed=None)
            valid_total_loss += metrics[1]
            valid_total_loss_l1 += metrics[0]
            valid_n_samples += metrics[2]
            if valid_output is None:
                valid_output = metrics[3]
                valid_label = y
            else:
                valid_output = torch.cat((valid_output, metrics[3]))
                valid_label = torch.cat((valid_label, y))

        valid_rse, valid_rae, valid_correlation = generate_metric(valid_total_loss, valid_total_loss_l1,
                                                valid_n_samples, data, valid_output, valid_label)

        engine.scheduler.step(valid_rse)

        s2 = time.time()
        # log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        val_time.append(s2 - s1)
        mtrain_loss = train_total_loss / n_samples

        mvalid_rse = valid_rse
        mvalid_rae = valid_rae
        mvalid_corr = valid_correlation
        his_loss.append(valid_rse.item())

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, ' \
              'Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        log_string(log.format(i, mtrain_loss,
                              mvalid_rse, mvalid_rae, mvalid_corr, (t2 - t1)))

        torch.save(engine.model.state_dict(),
                   args.save + "_epoch_" + str(i) + ".pth")

    log_string("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    log_string("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(args.save + "_epoch_" + str(bestid + 1) + ".pth"))
    engine.model.eval()
    outputs_r = []
    test_total_loss = 0
    test_total_loss_l1 = 0
    test_n_samples = 0
    test_predict = None
    test = None
    evaluatel2 = nn.MSELoss(size_average=False).to(device)
    evaluatel1 = nn.L1Loss(size_average=False).to(device)
    for iter, (x, y) in enumerate(data.get_batches(data.test[0], data.test[1], batch_size=args.batch_size, shuffle=False)):
        testx = x.unsqueeze(-1)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds, _, _ = engine.model(testx, pred_time_embed=None)
        preds = preds.squeeze()
        scale = data.scale.expand(preds.size(0), data.m)
        preds = preds * scale
        y = y * scale
        outputs_r.append(preds)

        test_total_loss += evaluatel2(preds, y).item()
        test_total_loss_l1 += evaluatel1(preds, y).item()
        test_n_samples += (preds.size(0) * data.m)

        if test_predict is None:
            test_predict = preds
            test = y
        else:
            test_predict = torch.cat((test_predict, preds))
            test = torch.cat((test, y))

    rse, rae, correlation = generate_metric(test_total_loss, test_total_loss_l1,
                                            test_n_samples, data, test_predict, test)

    log_string("The valid loss on best model is {}".format(str(round(his_loss[bestid], 4))))
    log_string('seed is {}'.format(args.seed))

    log = 'Evaluate best model on test data, Test rse: {:.4f}, Test rae: {:.4f}, Test corr: {:.4f}'
    log_string(log.format(rse, rae, correlation))

    torch.save(engine.model.state_dict(),
               args.save + "_exp" + str(args.expid) + "_best_" + str(args.order) + '_' + str(args.seed) + ".pth")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
