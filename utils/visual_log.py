import re
import string
import matplotlib.pyplot as plt
import numpy as np


color_map1 = {"PQ_full":"ro-", "IoU":"bo-", "Loss":"yo-", "Sem_l":"go-", "HP_l":"co-", 'Offset_l':"mo-"}
color_list = ["ro-", "bo-", "yo-","go-", "co-", "mo-"]


def get_l(log_path):
    with open(log_path,'r') as file:
        epoch = []
        PQ = []
        IoU = []
        Loss = []
        Sem_L = []
        HP_L = []
        Offset_L = []
        Instmap_L = []
        Lr = []
        miou_flag = 0
        for line in file:
            # PQ
            regex_pq = re.compile('Current val PQ is (\d+[.\d]*)')
            result_pq = re.findall(regex_pq, line)
            if result_pq:
                PQ.append(float(result_pq[0]))
                miou_flag = 1

            # mIoU
            if miou_flag == 1:
                regex_miou = re.compile('Current val miou is (\d+[.\d]*)')
                result_miou = re.findall(regex_miou, line)
                if result_miou:
                    IoU.append(float(result_miou[0]))
                    miou_flag = 0
            # loss
            regex_loss = re.compile('loss: (\d+[.\d]*), semantic loss:')
            result_loss = re.findall(regex_loss, line)
            if result_loss:
                Loss.append(float(result_loss[0]))
            
            # semantic loss
            regex_sem_loss = re.compile('semantic loss: (\d+[.\d]*),')
            sem_loss = re.findall(regex_sem_loss, line)
            if sem_loss:
                Sem_L.append(float(sem_loss[0]))

            # heatmap loss
            regex_hp_loss = re.compile('heatmap loss: (\d+[.\d]*),')
            hp_loss = re.findall(regex_hp_loss, line)
            if hp_loss:
                HP_L.append(float(hp_loss[0]))

            # offset loss
            regex_offset_loss = re.compile('offset loss: (\d+[.\d]*),')
            offset_loss = re.findall(regex_offset_loss, line)
            if offset_loss:
                Offset_L.append(float(offset_loss[0]))   

            # instmap loss
            regex_instmap_loss = re.compile('instmap loss: (\d+[.\d]*)')
            instmap_loss = re.findall(regex_instmap_loss, line)
            if instmap_loss:
                Instmap_L.append(float(instmap_loss[0]))     

            # lr
            regex_lr = re.compile('lr: (\d+[.\d]*)')
            result_lr = re.findall(regex_lr, line)
            if result_lr:
                Lr.append(float(result_lr[0]))
    file.close()
    return {
        'epoch': epoch,
        'PQ': PQ,
        'IoU': IoU,
        'Loss': Loss,
        'Sem_l': Sem_L,
        'HP_l': HP_L,
        'Offset_l': Offset_L,
        'Instmap_l': Instmap_L,
        'Lr': Lr
    }

def plot_l(Dict, mark_max=True, save_name=None):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for key, v in Dict.items():
        if key in ["PQ_full", "IoU"]:
            if v[0] == 0.0:
                v.pop(0)
            v = np.array(v)
            epoch = np.arange(1,len(v)+1)
            color = color_map1.get(key, "o-")
            ax1.plot(epoch, v, color , markersize=2.5, linewidth =1.2, label=str(key))
            if mark_max: #如果需要标出最大值
                xmax = np.argmax(v)
                plt.plot(xmax, v[xmax], 'ko')
                show_max = '[' + str(xmax) + ', ' + str(v[xmax]) + '%]'
                plt.annotate(show_max,xy=(xmax,v[xmax]), xytext=(xmax,v[xmax]))
    ax2 = ax1.twinx()
    for key, v in Dict.items():
        if key in ["Loss", "Sem_l", "HP_l, Offset_l", "Instmap_l", "Lr"]:
            if v[0] == 0.0:
                v.pop(0)
            v = np.array(v)
            epoch = np.arange(1,len(v)+1)
            color = color_map1.get(key, "o-")
            ax2.plot(epoch, v, color, markersize=2.5, linewidth =1.2, label=str(key))
            if mark_max: #如果需要标出最大值
                xmax = np.argmax(v)
                plt.plot(xmax, v[xmax], 'ko')
                show_max = '[' + str(xmax) + ', ' + str(v[xmax]) + ']'
                plt.annotate(show_max,xy=(xmax,v[xmax]),xytext=(xmax,v[xmax]))
        
    # plt.ylabel(k)
    # plt.title('Accuracy')
    ax1.set_xlabel('epoch')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    ax1.legend(h1, l1, loc='upper right')
    ax2.legend(h2, l2, loc='lower right')
    
    ax1.set_ylabel("PQ, mIoU")
    ax2.set_ylabel("All kinds of loss")
    
    
    plt.grid(True)  # 添加网格参考线，横纵都有
    # plt.grid(axis='y')  # 添加y轴网格线
    if save_name:
        plt.savefig(save_name)
    plt.show()

log_path = '../test_kitti_20230213_flip_2k2bs_3090.log'
# log_path = './pretrain_semi_nus_02fix_2bs3_20221227.log'
# log_path = '/data/cyj/Code/ppn69/nu_full_orinet.log'
l_Dict = get_l(log_path)
PQ_full = l_Dict['PQ']
# log_path0 = '../nohup_kitti_20230213_flip_2k2bs_3090.log'
# log_path1 = './checkpoint/plot_logs/kitti_pretrain/pretrain_04fix_kitti_pix_2bs1_20221214_useinstaug.log' #'./checkpoint/plot_logs/pretrain_semi_nus_04fix_2bs3_a6000_20221124.log'
# log_path2 = './checkpoint/plot_logs/kitti_pretrain/pretrain_02fix_kitti_pix_2bs1_20221223_useinstaug.log' #'./checkpoint/plot_logs/pretrain_semi_nus_flip_02fix_2bs3_20221220.log'
# log_path3 = './checkpoint/plot_logs/kitti_pretrain/pretrain_01fix_kitti_pix_2bs1_20221223_useinstaug.log' #'./checkpoint/plot_logs/pretrain_semi_nus_flip_01fix_2bs3_20221223.log'
# PQ_full = get_l(log_path0)['PQ']
# PQ_04fix = get_l(log_path1)['PQ']
# PQ_02fix = get_l(log_path2)['PQ']
# PQ_01fix = get_l(log_path3)['PQ']
Dict = {
    'PQ_full': PQ_full,
    'IoU': l_Dict['IoU'],
    'Loss': l_Dict['Loss'],
    'Sem_l': l_Dict['Sem_l'],
    'HP_l': l_Dict['HP_l'],
    'Offset_l': l_Dict['Offset_l'],
    'Instmap_l': l_Dict['Instmap_l'],
    'Lr': l_Dict['Lr']
}
plot_l(Dict, save_name='ppn68_is_aug2_kitti.png')
