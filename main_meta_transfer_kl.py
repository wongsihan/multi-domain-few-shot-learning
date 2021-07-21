import torch
import torch.nn as nn
from utlis.models4 import CNNEncoder1d, CNNEncoder2d, RelationNetwork1d, RelationNetwork2d
from network_M2L_KL import define_FewShotNet,accuracy
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import utlis.transfer_generator as tg
import argparse
import pickle
from utlis.logger import setlogger
import logging
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn.functional as F


parser = argparse.ArgumentParser(description="meta transfer learning")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=5)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=3)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=15)
parser.add_argument("-e", "--episode", type=int, default=100)
parser.add_argument("-t", "--test_episode", type=int, default=1000)
parser.add_argument("-l", "--learning_rate", type=float, default=0.01)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-d", "--datatype", type=str, default='fft')
parser.add_argument("-m", "--modeltype", type=str, default='1d')
parser.add_argument("-n", "--snr", type=int, default=-4)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
DATATYPE = args.datatype
MODELTYPE = args.modeltype
testgap = 5
SNR = args.snr


torch.set_default_tensor_type('torch.cuda.FloatTensor')
tg.set_seed(3)
# -----------------------------path---------------------------
root = '.\\result_transfer_mean'
path1 = DATATYPE + '_' + MODELTYPE + '_' + str(CLASS_NUM) + 'way' + str(SAMPLE_NUM_PER_CLASS) + 'shot'
path = os.path.join(root, path1)
if not os.path.exists("%s" % path):
    os.makedirs("%s" % path)

# ----------------------------logging-----------------------
setlogger(os.path.join("%s" % path, 'train.log'))
f = open("%s\\train.log" % path, 'w')
f.truncate()
f.close()


def main():
    # Step 1: init data folders
    logging.info("init data folders")
    # init character folders for dataset construction
    datapath = '.\\tempdata\\' + str(CLASS_NUM) + 'way' + '.pkl'
    if not os.path.exists("%s" % datapath):
        metatrain_character_folders, metatest_character_folders = tg.pu_folders(CLASS_NUM)
        os.makedirs('.\\tempdata')
        output = open(datapath, 'wb')
        pickle.dump((metatrain_character_folders, metatest_character_folders), output)
        output.close()
    else:
        with open(datapath, 'rb') as pkl:
            metatrain_character_folders, metatest_character_folders = pickle.load(pkl)

    # Step 2: init neural networks
    logging.info("init neural networks")

    if MODELTYPE == '1d':
        # feature_encoder = CNNEncoder1d(FEATURE_DIM)
        # relation_network = RelationNetwork1d(FEATURE_DIM, RELATION_DIM)
        feature_encoder = define_FewShotNet(which_model= 'Conv64F', num_classes=CLASS_NUM, neighbor_k=1,
                                         norm='batch',
                                         shot_num=SAMPLE_NUM_PER_CLASS, batch_size=SAMPLE_NUM_PER_CLASS*CLASS_NUM, init_type='normal',
                                         use_gpu=GPU)
    elif MODELTYPE == '2d':
        feature_encoder = CNNEncoder2d(FEATURE_DIM)
        relation_network = RelationNetwork2d(FEATURE_DIM, RELATION_DIM)


    feature_encoder.cuda(GPU)
    # relation_network.cuda(GPU)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=1000, gamma=0.5)
    # relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    # relation_network_scheduler = StepLR(relation_network_optim, step_size=1000, gamma=0.5)

    # Step 3: build graph
    logging.info("Training...")

    last_accuracy = 0.0
    accuracys = []
    aepochs = []
    losses = []
    lepochs = []
    finalsum = 0
    for episode in range(EPISODE):

        feature_encoder_scheduler.step(episode)
        # relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training

        task = tg.puTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False,
                                               dt=DATATYPE, mt=MODELTYPE)
        batch_dataloader = tg.get_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test", shuffle=False,
                                              dt=DATATYPE, mt=MODELTYPE,snr=SNR)

        # sample datas
        samples, sample_labels = sample_dataloader.__iter__().next()

        batches, batch_labels = batch_dataloader.__iter__().next()


        # calculate features
        #output, Q_S = feature_encoder(Variable(batches).cuda(GPU).float(),Variable(samples).cuda(GPU).float())  # 5x64*5*5
        output = feature_encoder(Variable(batches).cuda(GPU).float(),
                                      Variable(samples).cuda(GPU).float())  # 5x64*5*5


        criterion = nn.CrossEntropyLoss().cuda()
        loss = torch.tensor(0.).cuda()
        loss.requires_grad = True
        for i in range(len(output)):
            temp_loss = criterion(output[i], batch_labels.long())
            loss = loss + temp_loss

            # Measure accuracy and record loss
            prec1, _ = accuracy(output[i], batch_labels, topk=(1,3))
            # losses.update(temp_loss.item(), query_nums)
            # top1.update(prec1[0], query_nums)

        # Compute gradients and do SGD step
        feature_encoder_optim.zero_grad()
        loss.backward()
        feature_encoder_optim.step()

        logging.info("episode:" + str(episode + 1) + "   loss: " + str(loss.item())+ "  accuarcy: " + str(prec1.item()))
        losses.append(loss.item())
        lepochs.append(episode)

        if (episode + 1) % testgap == 0:

            # test
            logging.info("Testing...")
            total_rewards = 0

            for i in range(TEST_EPISODE):

                task = tg.puTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
                sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                       shuffle=False, dt=DATATYPE, mt=MODELTYPE)

                test_dataloader = tg.get_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test",
                                                     shuffle=True, dt=DATATYPE, mt=MODELTYPE,snr = SNR)

                sample_images, sample_labels = sample_dataloader.__iter__().next()
                test_images, test_labels = test_dataloader.__iter__().next()

                # calculate features
                #output, Q_S  = feature_encoder(Variable(test_images).cuda(GPU).float(),Variable(sample_images).cuda(GPU).float())  # 5x64
                output = feature_encoder(Variable(batches).cuda(GPU).float(),
                                              Variable(samples).cuda(GPU).float())  # 5x64*5*5

                prec1, _ = accuracy(output[0], test_labels, topk=(1,3))
                #_, predict_labels = torch.max(relations.data, 1)
                #predict_labels = predict_labels.int()

                # rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in
                #            range(CLASS_NUM * SAMPLE_NUM_PER_CLASS)]

                total_rewards += prec1

            test_accuracy = total_rewards / 1.0 /TEST_EPISODE

            logging.info("test accuracy:" + str(test_accuracy))

            # if episode+1 > 80:
            #     finalsum += test_accuracy

            accuracys.append(test_accuracy)
            aepochs.append(episode)

            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(), os.path.join(path, "feature_encoder_%f.pkl" % test_accuracy))
                logging.info("save networks for episode:" + str(episode))
                last_accuracy = test_accuracy
    torch.save(feature_encoder.state_dict(), os.path.join(path, "feature_encoder_final.pkl"))
    #torch.save(relation_network.state_dict(), os.path.join(path, "relation_network_final.pkl"))
    logging.info("final accuracy :" + str(test_accuracy))
    #logging.info("final finalsum :" + str(finalsum/4))
    plt.figure()
    plt.suptitle(path1)
    plt.subplot(2, 1, 1)
    plt.plot(aepochs, accuracys)
    plt.title('test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(lepochs, losses)
    plt.title('train loss')
    plt.savefig('%s\\accuracy.png' % path)
    output = open('%s\\accuracy.pkl' % path, 'wb')
    pickle.dump((aepochs, accuracys, lepochs, losses), output)
    output.close()


if __name__ == '__main__':

    main()
