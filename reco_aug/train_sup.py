import torch

import torchvision.models as models
import torch.utils.data.sampler as sampler
import torch.optim as optim
import argparse
import matplotlib.pylab as plt

from network.deeplabv3.deeplabv3 import *
from build_data import *
from module_list import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
import itertools
import io
from numpy import savetxt
import tensorflow as tf


parser = argparse.ArgumentParser(description='Supervised Segmentation with Perfect Labels')
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num_labels', default=15, type=int, help='number of labelled training data, set 0 to use all training data')
parser.add_argument('--lr', default=2.5e-3, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--dataset', default='cityscapes', type=str, help='pascal, cityscapes, sun')
parser.add_argument('--apply_reco', action='store_true')
parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
parser.add_argument('--output_dim', default=256, type=int, help='output dimension from representation head')
parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--strong_threshold', default=0.97, type=float)

args = parser.parse_args()

def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)
    
    return image

my_classes = ['bg', 'rigid_plastic', 'cardboard', 'metal', 'soft_plastic']
def conf_mat_to_png(conf_mat, out_name, data_name):
    np_cm = conf_mat.get_matrix()
    np_cm = np_cm.cpu().numpy()
    np_cm = np_cm.astype('float') / np_cm.sum(axis=1)[:, np.newaxis]
    savetxt(data_name, np_cm, delimiter=',')
    figg = plot_confusion_matrix(np_cm, my_classes)
    img = plot_to_image(figg)
    tf.keras.preprocessing.image.save_img(out_name,img[0])

def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

log_dir = create_directory(f'./logs/')
log_path = log_dir + 'sup_aug.txt'
log_func = lambda string='': log_print(string, log_path)


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data_loader = BuildDataLoader(args.dataset, args.num_labels)
train_l_loader, test_loader = data_loader.build(supervised=True)

# Loader Semantic Network
device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")
model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)

total_epoch = 15
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
scheduler = PolyLR(optimizer, total_epoch, power=0.9)
#create_folder(args.save_dir)

train_epoch = len(train_l_loader)
test_epoch = len(test_loader)
avg_cost = np.zeros((total_epoch, 6))
iteration = 0
for index in range(total_epoch):
    cost = np.zeros(3)
    train_l_dataset = iter(train_l_loader)

    model.train()
    conf_mat = ConfMatrix(data_loader.num_segments)
    for i in range(train_epoch):
        train_data, train_label = train_l_dataset.next()
        train_data, train_label = train_data.to(device), train_label.to(device)

        optimizer.zero_grad()

        pred, rep = model(train_data)
        pred_large = F.interpolate(pred, size=train_label.shape[1:], mode='bilinear', align_corners=True)
        sup_loss = compute_supervised_loss(pred_large, train_label)

        # regional contrastive loss
        if args.apply_reco:
            with torch.no_grad():
                mask = F.interpolate((train_label.unsqueeze(1) >= 0).float(), size=pred.shape[2:], mode='nearest')
                label = F.interpolate(label_onehot(train_label, data_loader.num_segments), size=pred.shape[2:], mode='nearest')
                prob = torch.softmax(pred, dim=1)

            reco_loss = compute_reco_loss(rep, label, mask, prob, args.strong_threshold, args.temp, args.num_queries, args.num_negatives)
            loss = sup_loss + reco_loss
        else:
            loss = sup_loss

        loss.backward()
        optimizer.step()

        # compute metrics by confusion matrix
        conf_mat.update(pred_large.argmax(1).flatten(), train_label.flatten())
        avg_cost[index, 0] += loss.item() / train_epoch

        iteration += 1

    avg_cost[index, 1:3] = conf_mat.get_metrics()

    np_cm = conf_mat.get_matrix()
    class_losses = np_cm.diag()/np_cm.sum(1)
    log_func('Train class loss:')
    log_func('bg: {}'.format(class_losses[0]))
    log_func('rigid_plastic: {}'.format(class_losses[1]))
    log_func('cardboard: {}'.format(class_losses[2]))
    log_func('metal: {}'.format(class_losses[3]))
    log_func('soft_plastic: {}'.format(class_losses[4]))
    conf_mat_to_png(conf_mat, 'sup_aug_train_conf_matrix.png', 'sup_aug_train_conf_matrix.csv')

    with torch.no_grad():
        model.eval()
        test_dataset = iter(test_loader)
        conf_mat = ConfMatrix(data_loader.num_segments)
        for i in range(test_epoch):
            test_data, test_label = test_dataset.next()
            test_data, test_label = test_data.to(device), test_label.to(device)

            pred, _ = model(test_data)
            pred_large = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)

            loss = compute_supervised_loss(pred_large, test_label)

            # compute metrics by confusion matrix
            conf_mat.update(pred_large.argmax(1).flatten(), test_label.flatten())
            avg_cost[index, 3:] += loss.item() / test_epoch
        avg_cost[index, 4:6] = conf_mat.get_metrics()

        np_cm = conf_mat.get_matrix()
        class_losses = np_cm.diag()/np_cm.sum(1)
        log_func('Train class loss:')
        log_func('bg: {}'.format(class_losses[0]))
        log_func('rigid_plastic: {}'.format(class_losses[1]))
        log_func('cardboard: {}'.format(class_losses[2]))
        log_func('metal: {}'.format(class_losses[3]))
        log_func('soft_plastic: {}'.format(class_losses[4]))
        conf_mat_to_png(conf_mat, 'sup_aug_val_conf_matrix.png', 'sup_aug_val_conf_matrix.csv')


    scheduler.step()
    """ print('EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f}'
          .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                  avg_cost[index][3], avg_cost[index][4], avg_cost[index][5]))
    print('Top: mIoU {:.4f} IoU {:.4f}'.format(avg_cost[:, 4].max(), avg_cost[:, 5].max())) """

    log_func('EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f} || Test [Loss | mIoU | Acc.]: {:.4f} {:.4f} {:.4f}'
          .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                  avg_cost[index][3], avg_cost[index][4], avg_cost[index][5]))

    log_func('Top: mIoU {:.4f} IoU {:.4f}'.format(avg_cost[:, 4].max(), avg_cost[:, 5].max()))

    log_func('\n')
    if avg_cost[index][4] >= avg_cost[:, 4].max():
        if args.apply_reco:
            torch.save(model.state_dict(), 'model_weights/{}_label{}_sup_reco_{}.pth'.format(args.dataset, args.num_labels, args.seed))
        else:
            torch.save(model.state_dict(), 'model_weights/{}_label{}_sup_{}.pth'.format(args.dataset, args.num_labels, args.seed))

    if args.apply_reco:
        np.save('logging/{}_label{}_sup_reco_{}.npy'.format(args.dataset, args.num_labels, args.seed), avg_cost)
    else:
        np.save('logging/{}_label{}_sup_{}.npy'.format(args.dataset, args.num_labels, args.seed), avg_cost)
