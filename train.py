import torch.optim as optim
import time
import torch
from torchvision import transforms
import torch.nn as nn
import pandas as pd 
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import sys
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from timm.scheduler import CosineLRScheduler
from timm.loss import *
from imagemodel import *

from dadaptation.dadapt_adam import DAdaptAdam


from ray import tune
from rw.timm.models.maxxvit import coatnet_1_224



## モデル
from imagemodel import *
from coatnet_convnext import CoAtNet,count_parameters,coatnet_1,coatnet_0,coatnet_2
# from coatnet_convnext_post import CoAtNet,count_parameters,coatnet_1,coatnet_0,coatnet_2
# from coatnet import coatnet_2, CoAtNet


# ###==========
# net = coatnet_1_224()
# net = timm.create_model(model_name="coatnet_1_224",pretrained=False)

###==========自分でCoAtNet組み立てるとき=======================
num_blocks = [2, 2, 6, 14, 2]           
channels = [64, 96, 192, 384, 768] 
block_types=['F','F','T','T']
net = CoAtNet((224, 224), 3, num_blocks, channels, block_types=block_types)



#==========torchvisionからモデルも持ってくる=======================
# from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
# model_image = efficientnet_b0(weight=None)
# model = timm.create_model("coatnet_1_rw_224", pretrained = True, num_classes = 4)



#torchvision.modelから
# from torchvision.models import resnet34,ResNet34_Weights
# net = resnet34(weights=None)



## 評価関数
from evaluation_function import fp,fn,tn,tp,accuracy,tpr,fpr



#コマンドライン引数をargsという変数に格納
args = sys.argv

#pathnumber = 2
pathnumber = args[1]

## パス データ拡張後
TRAIN_CSV_PATH = "csvfile_aug/notmask/"+str(pathnumber)+"train.csv"
VALIDATION_CSV_PATH = "csvfile_aug/notmask/"+str(pathnumber)+"test.csv"

TRAIN_IMAGEPATH = "./dataset/"         #画像格納先のパス
#===============================================================================================

## ハイパーパラメータ
Momentum = 0.9
Step_Size = 20
Gamma = 0.1
Lr_i = 6.4e-5    #  9.4e-5  
BATCHSIZE = 8

STEP_SIZE = 30

EPOCH = 50

#画像パスを保存
SAVE_IMAGE_PATH = './save_models/image_'+str(pathnumber)+'_.pth'

## グラフ名
image_Save_path_loss = './result_c1/FFTT/image_'+str(pathnumber)+'_loss.png'
image_Save_path_acc  = './result_c1/FFTT/image_'+str(pathnumber)+'_acc.png'

#********************************************************************
## テキストファイル名
textfilepath = './result_c1/FFTT/accuracy_file_train.txt'
#===============================================================================================
## 前処理
transform_dict = {
        'train': transforms.Compose(
            [transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), #Imagenetの標準化
            ]),
        'test': transforms.Compose(
            [transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ])}
#===============================================================================================






class MULTIMODALDATASET(Dataset):
    def __init__(self, csv_path,transform=None):
        ## csvファイル読み込み
        df = pd.read_csv(csv_path)

        image_paths = TRAIN_IMAGEPATH + df['filename']
        textlabel = np.array(df['index'])
        textlabel = torch.from_numpy(textlabel).long()

        self.image_paths = image_paths
        self.textlabel = textlabel
        self.transform = transform

    def __getitem__(self, index):
        path = self.image_paths[index] 
        ## 画像読み込み。
        img = Image.open(path)

        label=self.textlabel[index]     

        ## 事前処理実施
        if self.transform is not None:
            img = self.transform(img)

        return img,label

    ## データ数を返す
    def __len__(self):
        return len(self.image_paths)


## データセット定義
traindataset = MULTIMODALDATASET(TRAIN_CSV_PATH,transform=transform_dict["train"])
valdataset = MULTIMODALDATASET(VALIDATION_CSV_PATH,transform=transform_dict["test"])
## データローダー化
trainloader = DataLoader(traindataset, batch_size=BATCHSIZE, shuffle=True)
valloader = DataLoader(valdataset, batch_size=BATCHSIZE, shuffle=False)

## 辞書型変数にまとめる
train_size = int(len(traindataset))
val_size  = int(len(valdataset))  
data_size  = {"train":train_size, "val":val_size}
dataloaders  = {"train":trainloader, "val":valloader}

#===============================================================================================
## 画像特徴ベクトル抽出モデル

#＝＝＝＝＝＝＝＝＝ここを変更する＝＝＝＝＝＝＝＝＝＝＝(coatnet)

# image_model = IMAGE()
# image_dense_model = IMAGE_DENSE()


#===============================================================================================
## GPU使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#＝＝＝＝＝＝＝＝＝ここを変更する＝＝＝＝＝＝＝＝＝＝＝(coatnet)
# net = coatnet_2()
image_model = net.to(device)


# image_model = image_model.to(device)
# image_dense_model = image_dense_model.to(device)
#*****************************************************
## イメージ
## 損失関数　　## 交差エントロピー誤差

image_criterion = LabelSmoothingCrossEntropy(smoothing=0.1)  ##timm
# image_criterion = SoftTargetCrossEntropy()   ##timm
# image_criterion = nn.CrossEntropyLoss()

## 最適化手法　　## 確率的勾配降下法
# base_optimizer = torch.optim.SGD
# image_optimizer = SAM(image_model.parameters(), base_optimizer,lr=000.1, momentum=Momentum,weight_decay=4e-7)


# image_optimizer = SAMSGD(image_model.parameters(), lr=0.1,rho=0.05)
# image_optimizer = optim.SGD(image_model.parameters(), lr=Lr_i, momentum=Momentum ,weight_decay=4e-7)
# image_optimizer = optim.RAdam(image_model.parameters(),lr=Lr_i, weight_decay=4e-7)
# image_optimizer = optim.AdamW(image_model.parameters(),lr=Lr_i,weight_decay=0.001)
image_optimizer = optim.AdamW(image_model.parameters(),lr=Lr_i,weight_decay=0.005)
# image_optimizer = DAdaptAdam(params=image_model.parameters(),lr=1)
# params, lr=1.0, betas=(0.9, 0.999), eps=1e-8,weight_decay=0, log_every=0,decouple=False,d0=1e-6, growth_rate=float('inf') ##dadaptation デフォルトの設定





#学習率更新
scheduler = CosineLRScheduler(image_optimizer, t_initial=5, lr_min=1e-6)#, warmup_t=20, warmup_lr_init=5e-5, warmup_prefix=True)
# scheduler = optim.lr_scheduler.StepLR(image_optimizer, step_size=60, gamma=0.1)  #stepsizeごとに更新
# scheduler = optim.lr_scheduler.CyclicLR(image_optimizer,base_lr=0.000001,max_lr=0.0001,step_size_up=20,mode="exp_range",gamma=0.85)
# scheduler = optim.lr_scheduler.CyclicLR(image_optimizer,base_lr=0.00001,max_lr=0.0001,step_size_up=30,mode="exp_range",gamma=0.85)
# scheduler3 = optim.lr_scheduler.CyclicLR(image_optimizer,base_lr=0.0000001,max_lr=0.000001,step_size_up=10,mode="exp_range",gamma=0.85)
# scheduler = optim.lr_scheduler.CyclicLR(image_optimizer,base_lr=0.00001,max_lr=0.0001,step_size_up=10,mode="exp_range")
# scheduler = optim.lr_scheduler.MultiStepLR(image_optimizer,milestones=[20,40],gamma=0.1)  #milestoneの値ごとに更新
#*****************************************************
## 学習
def train_model(
    image_model,
    image_criterion, image_optimizer,
    num_epochs=EPOCH,
    ):
    since = time.time()

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True


    image_best_acc = 0.0
    best_loss = 1000.0
    best_epoc = 0

    # train_loss = 0.0
    val_loss = 0.0
    #=========  EPOCHここから  ============================================================
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 20)


        for phase in ['train', 'val']:
            if phase == 'train':
                ## 学習モード
                image_model.train() 
            else:
                ## 推論モード
                image_model.eval()  

            running_image_loss = 0.0
            running_image_corrects = 0
            

            AA = 0
            BB = 0
            CC = 0
            DD = 0
            EE = 0
            FF = 0
            GG = 0
            HH = 0
            II = 0
            JJ = 0
            KK = 0
            LL = 0
            MM = 0
            NN = 0
            OO = 0
            PP = 0

            #===========  イテレーションここから  ================================
            ## イテレーション
            for data in dataloaders[phase]:
                inputs ,labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                # # zero the parameter gradients
                image_optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    image_outputs = image_model(inputs)
                    

                    # image_dense_outputs = image_dense_model(image_outputs)
                    ##　ベクトルの中で最大値を返す
                    _, i_preds = torch.max(image_outputs, 1)

                    
                    
                    # _, i_preds = torch.max(image_dense_outputs, 1)

                    image_loss = image_criterion(image_outputs, labels)


                    ##SAM
                    # image_loss = image_criterion(image_outputs,labels)
                    # image_loss.backward()
                    # image_optimizer.first_step(zero_grad=True)

                    # image_criterion(image_outputs,labels).backward()
                    # image_optimizer.second_stop(zero_grad=True)
                    # image_loss = image_criterion(image_dense_outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #image
                        # zero the parameter gradients
                        # image_optimizer.zero_grad()
                        image_loss.backward()## 逆伝播計算
                        image_optimizer.step()## 勾配更新
                        
                        ##============================SAMを使うときだけ======================================
                        # image_loss = image_criterion(image_outputs, labels)
                        # image_loss.mean().backward()
                        # image_optimizer.first_step(zero_grad=True)
                        # image_criterion(image_outputs, labels).mean().backward()
                        # image_optimizer.second_step(zero_grad=True)
                    

                    if phase == 'val':

                        print(len(i_preds))
                        for i in range(len(i_preds)):
                            ## 正解ラベル0　
                            if labels[i]==0 and i_preds[i]==1:
                                print('BBB:{:.3f}'.format(i))
                            elif labels[i]==0 and i_preds[i]==2:
                                print('CCC:{:.3f}'.format(i))
                            elif labels[i]==0 and i_preds[i]==3:
                                print('DDD:{:.3f}'.format(i))
                            elif labels[i]==1 and i_preds[i]==0:
                                print('EEE:{:.3f}'.format(i))
                            elif labels[i]==2 and i_preds[i]==0:
                                print('III:{:.3f}'.format(i))
                            elif labels[i]==3 and i_preds[i]==0:
                                print('MMM:{:.3f}'.format(i))
                           
                            

                    
                    ## 画像
                    for i in range(len(i_preds)):
                        ## 正解ラベル0　
                        if labels[i]==0 and i_preds[i]==0:
                            AA+=1
                        elif labels[i]==0 and i_preds[i]==1:
                            BB+=1
                        elif labels[i]==0 and i_preds[i]==2:
                            CC+=1
                            # print('CC:{}'.format(i+1))
                        elif labels[i]==0 and i_preds[i]==3:
                            DD+=1
                            # print('DD:{}'.format(i+1))
                        ## 正解ラベル1
                        elif labels[i]==1 and i_preds[i]==0:
                            EE+=1
                        elif labels[i]==1 and i_preds[i]==1:
                            FF+=1
                        elif labels[i]==1 and i_preds[i]==2:
                            GG+=1
                        elif labels[i]==1 and i_preds[i]==3:
                            HH+=1
                        ##  正解ラベル2
                        elif labels[i]==2 and i_preds[i]==0:
                            II+=1
                        elif labels[i]==2 and i_preds[i]==1:
                            JJ+=1
                        elif labels[i]==2 and i_preds[i]==2:
                            KK+=1
                        elif labels[i]==2 and i_preds[i]==3:
                            LL+=1
                        ## 正解ラベル3
                        elif labels[i]==3 and i_preds[i]==0:
                            MM+=1
                            # print('MM:{}'.format(i+1))
                        elif labels[i]==3 and i_preds[i]==1:
                            NN+=1
                        elif labels[i]==3 and i_preds[i]==2:
                            OO+=1
                        elif labels[i]==3 and i_preds[i]==3:
                            PP+=1 

                ## image
                running_image_loss += image_loss.item() * inputs.size(0)
                running_image_corrects += torch.sum(i_preds == labels.data)

                epoch_image_loss = running_image_loss / data_size[phase]
                epoch_image_acc = running_image_corrects.double() / data_size[phase]


            #=========  イテレーションここまで  ================================
            ## image
            epoch_image_loss = running_image_loss / data_size[phase]
            epoch_image_acc = running_image_corrects.double() / data_size[phase]

            TP_i=tp(AA,FF,KK,PP)
            FP_i=fp(BB,CC,DD,EE,GG,HH,II,JJ,LL,MM,NN,OO)
            FN_i=fn(EE,II,MM,BB,JJ,NN,CC,GG,OO,DD,HH,LL)
            TN_i=tn(AA,BB,CC,DD,EE,FF,GG,HH,II,JJ,KK,LL,MM,NN,OO,PP)

            ACCURACY_i = accuracy(TP_i,data_size[phase])
            # PRECISION_i = precision(TP_i,FP_i)
            # RECALL_i = recall(TP_i,FN_i)
            TPR_i =  tpr(TP_i,FN_i)
            FPR_i =  fpr(FP_i,TN_i)

            # if phase == 'val':
            #     ## image
            #     print('　image　予測')
            #     print('　AA:{},BB:{},CC:{},DD:{}'.format(AA,BB,CC,DD))
            #     print('　EE:{},FF:{},GG:{},HH:{}'.format(EE,FF,GG,HH))
            #     print('　II:{},JJ:{},KK:{},LL:{}'.format(II,JJ,KK,LL))
            #     print('　MM:{},NN:{},OO:{},PP:{}'.format(MM,NN,OO,PP))

            
                # 　　　　　予測　　　　
                #     ------------------
                # 実　　 |[[a,b,c,d]
                # 際     | e,f,g,h
                #     　 | i,j,k,l
                #     　 | m,n,o,p]] 
                
                
            if (epoch+1) % 10 == 0 and phase == 'val':
                if epoch+1==10:
                    tentime = time.time() - since
                    nowtime = time.time() - since

                print('~'*10)
                print('multimodal')
                print('~'*10)
                print('ACCURACY:{:.3f}'.format(ACCURACY_i))

                # print('PRECISION:{:.3f}'.format(PRECISION))
                # print('RECALL:{:.3f}'.format(RECALL))
                print('TPR:{:.3f}'.format(TPR_i))
                print('FPR:{:.3f}'.format(FPR_i))

                print('best_now{:.3f}'.format(image_best_loss_acc))
                print('best_epoc:{:.3f}'.format(best_epoc))
                now = ((tentime+nowtime)/2)*(EPOCH-epoch-1)//10
                print('推定残り時間: {:.0f}分 {:.0f}秒'.format(
                        now // 60, now % 60))
                
                #======================学習率更新するとき(学習率の確認)=================================
            
                ###======================公式のschedulerのみ============================
                # print('現在の学習率:','epoch:{}, lr:{}'.format(epoch+1, scheduler.get_last_lr()[0]))

            ##  グラフをプロットするようにリストに格納
            if phase == 'train':
                image_train_loss_list.append(epoch_image_loss)
                image_train_acc_list.append(ACCURACY_i)
                # train_loss = epoch_image_loss

            else:
                image_val_loss_list.append(epoch_image_loss)
                image_val_acc_list.append(ACCURACY_i)
                val_loss = epoch_image_loss

            ## 一番高い正解率を出す


            #ここはなくてもいいみたい
            if phase == 'val' and ACCURACY_i > image_best_acc:  
                image_best_acc = ACCURACY_i
                image_best_tpr = TPR_i
                image_best_fpr = FPR_i

            #ロスがより小さいときに更新
            if phase == 'val' and  epoch_image_loss < best_loss:
                best_loss = epoch_image_loss
                best_epoc = epoch

                a = AA
                b = BB
                c = CC
                d = DD
                e = EE
                f1 = FF
                g = GG
                h = HH
                i = II
                j = JJ
                k = KK
                l = LL
                m = MM
                n = NN
                o = OO
                p = PP

                print('image　予測')
                print('AA:{},BB:{},CC:{},DD:{}'.format(a,b,c,d))
                print('EE:{},FF:{},GG:{},HH:{}'.format(e,f1,g,h))
                print('II:{},JJ:{},KK:{},LL:{}'.format(i,j,k,l))
                print('MM:{},NN:{},OO:{},PP:{}'.format(m,n,o,p))
                

                image_best_loss_acc = ACCURACY_i
                image_best_loss_tpr = TPR_i
                image_best_loss_fpr = FPR_i
                torch.save(image_model.state_dict(), SAVE_IMAGE_PATH)
                
            # if phase == 'train':
            #     if epoch_image_loss < 0.575:
            #         scheduler.step(epoch)   #学習率更新
    #=========  EPOCHここまで  ============================================================
        print()

    time_elapsed = time.time() - since

    print('**'*15)
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val ACCURACY: {:.3f} Best TPR:{:.3f} Best FPR:{:.3f}'.format(image_best_acc,image_best_tpr,image_best_fpr))
    print('Best TPR:{:.3f}'.format(image_best_tpr))
    print('Best FPR:{:.3f}'.format(image_best_fpr))
    print('Best val loss ACCURACY: {:.3f} Best loss TPR:{:.3f} Best loss FPR:{:.3f}'.format(image_best_loss_acc,image_best_loss_tpr,image_best_loss_fpr))

    print('BEST LOSS : ',best_loss)
    # print("パラメータ数：",count_parameters(net))
    print('**'*15)


    ## グラフ描画
    image_graph(EPOCH,image_train_loss_list,image_train_acc_list,image_val_loss_list,image_val_acc_list,image_Save_path_loss,image_Save_path_acc)


    

    return image_best_loss_acc,image_best_loss_tpr,image_best_loss_fpr,best_epoc,a,b,c,d,e,f1,g,h,i,j,k,l,m,n,o,p

## グラフ描画
def image_graph(epoch,trainloss,trainacc,valloss,valacc,loss,acc):
    plt.figure()
    plt.plot(range(epoch), trainloss,label = 'train_loss')
    plt.plot(range(epoch), valloss,label = 'val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(str(pathnumber)+'-image-loss')
    plt.savefig(loss)
    plt.grid()

    plt.figure()
    plt.plot(range(epoch), trainacc,label = 'train_acc')
    plt.plot(range(epoch), valacc,label = 'val_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.title(str(pathnumber)+'-image-accuracy')
    plt.savefig(acc)
    plt.grid()

## グラフ用のリスト
image_train_loss_list = []
image_train_acc_list = []
image_val_loss_list = []
image_val_acc_list = []


if __name__ == '__main__':

    print('='*20+str(pathnumber)+'回目'+'='*20)
    # main
    # textfile
    image_best_loss_acc,image_best_loss_tpr,image_best_loss_fpr,best_epoc,a,b,c,d,e,f1,g,h,i,j,k,l,m,n,o,p = train_model(
        image_model,
        image_criterion, image_optimizer,num_epochs=EPOCH
        )
    with open(textfilepath, mode='a') as f:
        f.write("*"*20+"\n")
        f.write("train\n")
        f.write("Epoch is "+str(EPOCH)+"\n")
        f.write("Batchsize is "+str(BATCHSIZE)+"\n")
        f.write("Learing rate is "+str(Lr_i)+"\n")
        
        f.write("Datanumber is "+str(pathnumber)+':: best_epoc = {:.3f}\n'.format(best_epoc))
        f.write("Datanumber is "+str(pathnumber)+':: image_accuracy = {:.3f}\n'.format(image_best_loss_acc))
        f.write("Datanumber is "+str(pathnumber)+':: image_tpr = {:.3f}\n'.format(image_best_loss_tpr))
        f.write("Datanumber is "+str(pathnumber)+':: image_fpr = {:.3f}\n\n'.format(image_best_loss_fpr))

        f.write("Datanumber is "+str(pathnumber)+':: AA:{},BB:{},CC:{},DD:{}\n'.format(a,b,c,d))
        f.write("Datanumber is "+str(pathnumber)+':: EE:{},FF:{},GG:{},HH:{}\n'.format(e,f1,g,h))
        f.write("Datanumber is "+str(pathnumber)+':: II:{},JJ:{},KK:{},LL:{}\n'.format(i,j,k,l))
        f.write("Datanumber is "+str(pathnumber)+':: MM:{},NN:{},OO:{},PP:{}\n'.format(m,n,o,p))

        f.write("*"*20+"\n")


    print()
    print()
