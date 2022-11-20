import time

from torch.optim import optimizer
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from dataset_loader import load_dataset
from CoATR import *
import urllib.request
import zipfile
import numpy as np
import torch
from collections import defaultdict
import argparse
import os
from torch.utils.tensorboard import SummaryWriter   
import scipy.io

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def seed_torch(seed=1029):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = True


def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_pred - y_true))

# def mape(y_true, y_pred, threshold=0.1):
#     v = torch.clip(torch.abs(y_true), threshold, None)
#     diff = torch.abs((y_true - y_pred) / v)
#     return 100.0 * torch.mean(diff, axis=-1).mean()

def rse(y_true, y_pred):
    batch_num, j = y_true.shape
    return torch.sqrt(torch.square(y_pred - y_true).sum()/(batch_num-2))

def mape(var, var_hat):
    return torch.sum(torch.abs(var - var_hat) / var) / var.shape[0]

def Print_loss(name, RMES, MAE, MAPE, RSE, m):
    print("%s_dataset sum loss: RMSE:%f, MAE:%f, MAPE:%f, RSE:%f" %(name, RMES, MAE/m, MAPE/m, RSE/m))



class TimeRecorder(object):
    def __init__(self):
        self.timer = dict(dataloader=0.0000001, train=0.0000001, test=0.0000001)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

if __name__ == "__main__":
    seed_torch(42)
    writter = SummaryWriter("./events")
    parser = argparse.ArgumentParser()
    parser.add_argument("--Batch_size", type=int, default=4, nargs="?", help="Batch size.")
    parser.add_argument("--epochs", type=int, default=250, nargs="?", help="epochs")
    parser.add_argument("--nc", type=int, default=180, nargs="?", help="Channel")
    parser.add_argument("--TR_ranks", type=int, default=40, nargs="?", help="TR-ranks")
    parser.add_argument("--d", type=int, default=40, nargs="?", help="AR")
    parser.add_argument("--Lambda", type=float, default=0.5, nargs="?", help="Hyperparameter")
    parser.add_argument("--mr", type=float, default=0.80, nargs="?", help="Missing rate")
    parser.add_argument("--input_dropout", type=float, default=0.26694419227220374, nargs="?", help="Input layer dropout.")
    parser.add_argument("--hidden_dropout", type=float, default=0.20, nargs="?", help="Hidden layer dropout.")

    args = parser.parse_args()
    kwargs = {'input_dropout': args.input_dropout, 'hidden_dropout': args.hidden_dropout, 'Batch_size': args.Batch_size,'AR_D': args.d}



    shape = torch.tensor((323, 28, 288)) # Seattle
    data_train, data_test, original_data = load_dataset('Traffic/Seattle.mat', args.mr)


    def worker_seed_fn(worker_id):
        return seed_torch(42 + worker_id + 1)
    
    # DataLoader
    loader_train = DataLoader(dataset=data_train, batch_size=args.Batch_size,
                                  shuffle=True, num_workers=4, worker_init_fn=worker_seed_fn,
                                  pin_memory=True, persistent_workers=True
                                  )
    loader_test = DataLoader(dataset=data_test, batch_size=args.Batch_size,
                                 shuffle=False, num_workers=4, worker_init_fn=worker_seed_fn,
                                 pin_memory=True, persistent_workers=True)

    model = TRDnet(shape, args.TR_ranks, args.nc, **kwargs) 
    model = model.cuda() 

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mseloss = torch.nn.MSELoss().cuda()
    loss_train=list()
    loss_test=list()
    timeRecord = TimeRecorder()

    index_train = torch.from_numpy(data_train.idx).cuda()
    index_train = index_train.view(-1, len(shape))
    labels_train = torch.from_numpy(data_train.label).cuda().view(-1, 1)
    index_true_train = torch.from_numpy(data_train.idx_true).cuda().view(-1, 1)

    RMSE_Count = 999
    MAPE_Count = 999
    Pred_Count = torch.tensor((1))


    for i in range(0, args.epochs):

        rmse_tr, mae_tr, mape_tr, rse_tr, lossA_tr= 0, 0, 0, 0, 0  
        rmse_te, mae_te, mape_te, rse_te, lossA_te= 0, 0, 0, 0, 0

        model.train()
        timeRecord.record_time()
        dataloader_time = 0.0
        train_time = 0.0
        print('==================Train/Test begin==================')
        print("Loss type: %s, Missing Rate:%.2f" %("NM", args.mr))
        for idex, label in loader_train: 
            i_shape = idex.shape
            idex = idex.view(-1,shape.shape[0]).cuda() 
            label = label.cuda() 
            dataloader_time += timeRecord.split_time()
            
            out, AR_loss = model(idex, args.TR_ranks, i_shape[0])
            out = torch.reshape(out, (-1, 1))

            label = torch.reshape(label, (-1, 1))
            label = label.to(torch.float32)
            TR_loss_train = torch.sqrt(mseloss(out, label))
            loss = TR_loss_train +  args.Lambda * AR_loss
            

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_time += timeRecord.split_time()

            rmse_tr += TR_loss_train.item()
            mae_tr += mae(label, out).item()
            mape_tr += mape(label, out).item()
            rse_tr += rse(label, out).item()
            
        rmse_tr = rmse_tr/len(loader_train)
        writter.add_scalar("RMSE_tr", rmse_tr, global_step=i)
        Print_loss('Train',rmse_tr,mae_tr,mape_tr, rse_tr,len(loader_train))
 

        # if (i % 5) == 0: 
        #     # torch.save(model.state_dict(), "./out/model_2.pyt")
        #     torch.save(model, "./out/my_model.pyt")

        timeRecord.record_time()

        with torch.no_grad():
            model.eval()
            index_list_test = []
            label_list_test = []
            pred_test = []
            for idex_2, label_2 in loader_test:
                i_shape = idex_2.shape
                idex_2 = idex_2.view(-1,shape.shape[0]).cuda() 
                # idex_2 = idex_2.cuda() 
                index_list_test.append(idex_2)
                label_2 = label_2.cuda() 
                
                
                out_2, re_loss_2 = model(idex_2, args.TR_ranks, i_shape[0])
                # out, AR_loss = model(idex, args.TR_ranks, args.Batch_size)

                out_2 = torch.reshape(out_2, (-1, 1))
                pred_test.append(out_2)
                label_2 = torch.reshape(label_2, (-1, 1))
                label_list_test.append(label_2)
                label_2 = label_2.to(torch.float32)
                loss_2 = torch.sqrt(mseloss(out_2, label_2)) 

                
                rmse_te += loss_2.item()
                mae_te += mae(label_2, out_2).item()
                mape_te += mape(label_2, out_2).item()
                rse_te += rse(label_2, out_2).item()

            rmse_te = rmse_te/len(loader_test)
            Print_loss('Test',rmse_te, mae_te, mape_te, rse_te, len(loader_test))
            loss_test.append(round(rmse_te,4 ))
            writter.add_scalar("RMSE_te", rmse_te, global_step=i)


            # Reconstruction loss
            index_tensor_test = torch.cat(index_list_test, dim=0)
            label_list_test= torch.cat(label_list_test,dim=0)
            pred_test = torch.cat(pred_test, dim=0)

            index_full = torch.cat((index_train, index_tensor_test), dim=0)
            labels = torch.cat((labels_train, pred_test), dim=0)

            index_true_test = torch.from_numpy(data_test.idx_true).cuda().view(-1, 1)
            index_true = torch.cat((index_true_train, index_true_test), dim=0)
            index_sort = torch.argsort(index_true, dim=0)
            index_full = index_full[index_sort.squeeze(-1)]
            labels = labels[index_sort.squeeze(-1)]

            original_data = torch.reshape(original_data, (-1, 1)).cuda()
            Rec_loss = torch.sqrt(mseloss(labels, original_data)) 
            test_loss = torch.sqrt(mseloss(pred_test, label_list_test))
            test_mape = mape(pred_test, label_list_test)


            # print(Rec_loss)
            # print("%s_dataset sum Reconstruction loss: RMSE:%f,loss_Conv:%f, MAE:%f, MAPE:%f, RSE:%f" %(name, RMES,lossA, MAE/m, MAPE/m, RSE/m))
            print("Dataset sum Reconstruction loss: RMSE:%f" % (Rec_loss))
            print("Dataset test Reconstruction loss: RMSE:%f" % (test_loss))
            print("Dataset test Reconstruction loss: MAPE:%f" % (test_mape))


        if (test_loss <= RMSE_Count):
            RMSE_Count = test_loss
            MAPE_Count = test_mape
            # torch.save(model, "./out/GZ0.8_NM_r40_nc128_h20_.pyt")
            print("********Data update********")
            Pred_Count = labels

        print("The best RMSE is %f, MAPE is %f" % (RMSE_Count, MAPE_Count))


        test_time = timeRecord.split_time()

        print('-------------Epoch Time Statistics-------------')

        print(f'epoch: {i + 1}, dataloader_time: {dataloader_time}, train_time: {train_time}, test_time: {train_time}')

        timeRecord.timer['dataloader'] += dataloader_time
        timeRecord.timer['train'] += train_time
        timeRecord.timer['test'] += test_time

        print("epoch:", i+1, "end")

    Pred_Count = torch.reshape(Pred_Count,(shape[0],shape[1],shape[2]))
    Pred_Count = Pred_Count.cpu().detach().numpy()
    scipy.io.savemat('./result_'+ str(args.mr)+"_"+ str(args.TR_ranks)+ "_"+ str(args.d)+ ".mat", mdict={'out': Pred_Count})

    print('==================Statistical Results==================')
    print('-------------Sum Time Statistics-------------')

    print(f'sum dataloader time: {timeRecord.timer["dataloader"]},sum train time: {timeRecord.timer["train"]}, sum test time: {timeRecord.timer["test"]}')

    print('-------------Loss Statistics-------------')
    print("loss_test:" , loss_test)  
    print("loss_test_total:" , loss_train)



