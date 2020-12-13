import argparse
import tensorflow as tf
import os
import sys
import time
import yaml

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from voc_data import create_batch_generator
from anchor import generate_default_boxes
from network import create_ssd
from losses import create_losses


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data/VOCdevkit')
parser.add_argument('--data-year', default='2012')
parser.add_argument('--arch', default='ssd300')
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--num-batches', default=-1, type=int)
parser.add_argument('--neg-ratio', default=3, type=int)
parser.add_argument('--initial-lr', default=1e-4, type=float)
#parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-5, type=float)
parser.add_argument('--num-epochs', default=100, type=int)
parser.add_argument('--checkpoint-dir', default='checkpoints')
parser.add_argument('--pretrained-type', default='base')
parser.add_argument('--gpu-id', default='0')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

NUM_CLASSES = 6


@tf.function
def train_step(imgs, gt_confs, gt_locs, ssd, criterion, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = ssd(imgs)

        conf_loss, loc_loss = criterion(
            confs, locs, gt_confs, gt_locs)

        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in ssd.trainable_variables]
        l2_loss = args.weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss

    gradients = tape.gradient(loss, ssd.trainable_variables)
    optimizer.apply_gradients(zip(gradients, ssd.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


if __name__ == '__main__':
    ## WJ ## Because of GPU Sync fail
    '''
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.compat.v1.Session(config=config)
    session.close()
    '''


    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open('./config.yml') as f:
        cfg = yaml.load(f)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError('Unknown architecture: {}'.format(args.arch))

    default_boxes = generate_default_boxes(config)

    batch_generator, val_generator, info = create_batch_generator(
        args.data_dir, args.data_year, default_boxes,
        config['image_size'],
        args.batch_size, args.num_batches,
        mode='train', augmentation=['patch', 'flip']) #, 'brightness', 'contrast', 'hue', 'saturation'
    try:
        ssd = create_ssd(NUM_CLASSES, args.arch,
                        args.pretrained_type,
                        checkpoint_dir=args.checkpoint_dir)
    except Exception as e:
        print(e)
        print('The program is exiting...')
        sys.exit()

    criterion = create_losses(args.neg_ratio, NUM_CLASSES)

    steps_per_epoch = info['length'] // args.batch_size

    lr_fn = PiecewiseConstantDecay(
        boundaries=[int(steps_per_epoch * args.num_epochs * 2 / 3),
                    int(steps_per_epoch * args.num_epochs * 5 / 6)],
        values=[args.initial_lr, args.initial_lr * 0.1, args.initial_lr * 0.01])
    
#    optimizer = tf.keras.optimizers.SGD(
#        learning_rate=lr_fn,
#        momentum=args.momentum)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_fn)

    train_log_dir = 'logs/train'
    val_log_dir = 'logs/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)


    ## WJ ##
    value_list = []


    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):
            loss, conf_loss, loc_loss, l2_loss = train_step(
                imgs, gt_confs, gt_locs, ssd, criterion, optimizer)
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
            if (i + 1) % 10 == 0:
                print('Epoch: {} Step {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}'.format(
                    epoch + 1, i + 1, time.time() - start, avg_loss, avg_conf_loss, avg_loc_loss))

        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0
        for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
            val_confs, val_locs = ssd(imgs)
            val_conf_loss, val_loc_loss = criterion(
                val_confs, val_locs, gt_confs, gt_locs)
            val_loss = val_conf_loss + val_loc_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (i + 1)
            avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)
        print('Epoch: {} | val_Loss: {:.4f} val_Conf: {:.4f} val_Loc: {:.4f}'.format(
                    epoch + 1, avg_val_loss, avg_val_conf_loss, avg_val_loc_loss))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', avg_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_loc_loss, step=epoch)

        with val_summary_writer.as_default():
            tf.summary.scalar('loss', avg_val_loss, step=epoch)
            tf.summary.scalar('conf_loss', avg_val_conf_loss, step=epoch)
            tf.summary.scalar('loc_loss', avg_val_loc_loss, step=epoch)
        
        
        ## WJ ##
        value_list.append([epoch +1, avg_loss, avg_conf_loss, avg_loc_loss, avg_val_loss, avg_val_conf_loss, avg_val_loc_loss])
        with open('log.txt', 'w') as f:
            for value in value_list:
                for word in value:
                    f.write(str(word) + " ")
                f.write("\n")
        
        if (epoch + 1) % 10 == 0:
            ssd.save_weights(
                os.path.join(args.checkpoint_dir, 'ssd_epoch_{}.h5'.format(epoch + 1)))
