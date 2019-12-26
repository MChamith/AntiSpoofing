import os
# os.environ['CUDA_VISIBLE_DEVICES'] =  '6'
import sys

from numpy import mean

from process.data_helper import submission
import cv2

sys.path.append("..")
import argparse
from process.data_fusion import *
from process.augmentation import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart


def get_model(model_name, num_class):
    if model_name == 'baseline':
        from model_fusion.model_baseline_SEFusion import FusionNet
    elif model_name == 'model_A':
        from model_fusion.FaceBagNet_model_A_SEFusion import FusionNet
    elif model_name == 'model_B':
        from model_fusion.FaceBagNet_model_B_SEFusion import FusionNet

    net = FusionNet(num_class=num_class)
    return net


def run_train(config):
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir, config.model_name)
    initial_checkpoint = config.pretrained_model
    criterion = softmax_cross_entropy_criterion

    ## setup  -----------------------------------------------------------------------------
    if not os.path.exists(out_dir + '/checkpoint'):
        os.makedirs(out_dir + '/checkpoint')
    if not os.path.exists(out_dir + '/backup'):
        os.makedirs(out_dir + '/backup')
    if not os.path.exists(out_dir + '/backup'):
        os.makedirs(out_dir + '/backup')

    log = Logger()
    log.open(os.path.join(out_dir, config.model_name + '.txt'), mode='a')
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    train_dataset = FDDataset(mode='train', modality=config.image_mode, image_size=config.image_size,
                              fold_index=config.train_fold_index)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=config.batch_size,
                              drop_last=True,
                              num_workers=8)

    valid_dataset = FDDataset(mode='val', modality=config.image_mode, image_size=config.image_size,
                              fold_index=config.train_fold_index)
    valid_loader = DataLoader(valid_dataset,
                              shuffle=False,
                              batch_size=config.batch_size // 36,
                              drop_last=False,
                              num_workers=8)

    assert (len(train_dataset) >= config.batch_size)
    log.write('batch_size = %d\n' % (config.batch_size))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')
    log.write('** net setting **\n')

    net = get_model(model_name=config.model, num_class=2)
    print(net)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir + '/checkpoint', initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n' % (type(net)))
    log.write('criterion=%s\n' % criterion)
    log.write('\n')

    iter_smooth = 20
    start_iter = 0
    log.write('\n')

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write(
        '                                  |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write(
        'model_name   lr   iter  epoch     |     loss      acer      acc    |     loss              acc     |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    train_loss = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)
    batch_loss = np.zeros(6, np.float32)
    iter = 0
    i = 0

    start = timer()
    # -----------------------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0005)

    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.cycle_inter,
                                          T_mult=1,
                                          model=net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=1e-3)

    global_min_acer = 1.0
    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))
        min_acer = 1.0

        for epoch in range(0, config.cycle_inter):
            sgdr.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.4f}'.format(lr))

            sum_train_loss = np.zeros(6, np.float32)
            sum = 0
            optimizer.zero_grad()

            for input, truth in train_loader:
                iter = i + start_iter
                # one iteration update  -------------
                net.train()
                input = input.cuda()
                truth = truth.cuda()

                logit, _, _ = net.forward(input)
                truth = truth.view(logit.shape[0])

                loss = criterion(logit, truth)
                precision, _ = metric(logit, truth)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:2] = np.array((loss.item(), precision.item(),))
                sum += 1
                if iter % iter_smooth == 0:
                    train_loss = sum_train_loss / sum
                    sum = 0

                i = i + 1

            if epoch >= config.cycle_inter // 2:
                # if 1:
                net.eval()
                valid_loss, _ = do_valid_test(net, valid_loader, criterion)
                net.train()

                if valid_loss[1] < min_acer and epoch > 0:
                    min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')

                if valid_loss[1] < global_min_acer and epoch > 0:
                    global_min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/global_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save global min acer model: ' + str(min_acer) + '\n')

            asterisk = ' '
            log.write(
                config.model_name + ' Cycle %d: %0.4f %5.1f %6.1f | %0.6f  %0.6f  %0.3f %s  | %0.6f  %0.6f |%s \n' % (
                    cycle_index, lr, iter, epoch,
                    valid_loss[0], valid_loss[1], valid_loss[2], asterisk,
                    batch_loss[0], batch_loss[1],
                    time_to_str((timer() - start), 'min')))

        ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_final_model.pth'
        torch.save(net.state_dict(), ckpt_name)
        log.write('save cycle ' + str(cycle_index) + ' final model \n')


def run_test(config, dir):
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = './models'
    out_dir = os.path.join(out_dir, config.model_name)
    initial_checkpoint = config.pretrained_model

    ## net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir + '/checkpoint', initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))

    valid_dataset = FDDataset(mode='val', modality=config.image_mode, image_size=config.image_size,
                              fold_index=config.train_fold_index)
    valid_loader = DataLoader(valid_dataset,
                              shuffle=False,
                              batch_size=1,
                              drop_last=False,
                              num_workers=8)

    test_dataset = FDDataset(mode='test', modality=config.image_mode, image_size=config.image_size,
                             fold_index=config.train_fold_index)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=config.batch_size,
                             drop_last=False,
                             num_workers=8)

    criterion = softmax_cross_entropy_criterion
    net.eval()

    # valid_loss, out = do_valid_test(net, valid_loader, criterion)
    # print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))

    print('infer!!!!!!!!!')
    filename = '/home/ubuntu/volume/data/IMG_1800.MOV'
    cap = cv2.VideoCapture(filename)
    # face_cascade = cv2.CascadeClassifier(
    #     '/home/chamith/Documents/Project/msid_server/venv/lib/python3.6/site-packages/cv2/data'
    #     '/haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # print('faces ' + str(faces))
        if faces is not None:
            # if True:
            for face in faces:
                # if True:
                x = face[0]
                y = face[1]
                w = face[2]
                h = face[3]
                color = frame[y:y + h, x:x + w]
                # print('color is' + str(color))
        # color = cv2.imrea d(os.path.join(DATA_ROOT, color), 1)
                depth = color.copy()
                ir = color.copy()

                color = cv2.resize(color, (RESIZE_SIZE, RESIZE_SIZE))
                depth = cv2.resize(depth, (RESIZE_SIZE, RESIZE_SIZE))
                ir = cv2.resize(ir, (RESIZE_SIZE, RESIZE_SIZE))

                # if self.mode == 'train':
                #     # print('color shape before augmentor ' + str(color.shape))
                #     ycrcb = cv2.cvtColor(depth, cv2.COLOR_BGR2YCR_CB)
                #     hsv = cv2.cvtColor(ir, cv2.COLOR_BGR2HSV)
                #     color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3))
                #     ycrcb = color_augumentor(ycrcb, target_shape=(self.image_size, self.image_size, 3))
                #     hsv = color_augumentor(hsv, target_shape=(self.image_size, self.image_size, 3))
                #
                #     # print('color shape ' + str(color.shape))
                #     # color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                #
                #
                #     # color = cv2.resize(color, (self.image_size, self.image_size))
                #     # ycrcblbp = cv2.resize(ycrcblbp, (self.image_size, self.image_size))
                #     # lbp = cv2.resize(lbp, (self.image_size, self.image_size))
                #
                #     image = np.concatenate([color.reshape([self.image_size, self.image_size, 3]),
                #                             ycrcb.reshape([self.image_size, self.image_size, 3]),
                #                             hsv.reshape([self.image_size, self.image_size, 3])],
                #                            axis=2)
                #
                #     if random.randint(0, 1) == 0:
                #         random_pos = random.randint(0, 2)
                #         if random.randint(0, 1) == 0:
                #             image[:, :, 3 * random_pos:3 * (random_pos + 1)] = 0
                #         else:
                #             for i in range(3):
                #                 if i != random_pos:
                #                     image[:, :, 3 * i:3 * (i + 1)] = 0
                #
                #     image = np.transpose(image, (2, 0, 1))
                #     image = image.astype(np.float32)
                #     image = image.reshape([self.channels * 3, self.image_size, self.image_size])
                #     image = image / 255.0
                #
                #     label = int(label)
                #     return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))
                #
                # elif self.mode == 'val':
                #     ycrcb = cv2.cvtColor(depth, cv2.COLOR_BGR2YCR_CB)
                #     hsv = cv2.cvtColor(ir, cv2.COLOR_BGR2HSV)
                #
                #     color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
                #     ycrcb = color_augumentor(ycrcb, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
                #     hsv = color_augumentor(hsv, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
                #     n = len(color)
                #
                #     color = np.concatenate(color, axis=0)
                #     ycrcb = np.concatenate(ycrcb, axis=0)
                #     hsv = np.concatenate(hsv, axis=0)
                #     # cv2.imwrite('color.jpg', color)
                #     # cv2.imwrite('depth.jpg', depth)
                #     # cv2.imwrite('ir.jpg', ir)
                #     # print('color sh/ape ' + str(type(depth)))
                #     # print('depth shape' + str(depth.shape))
                #
                #     image = np.concatenate([color.reshape([n, self.image_size, self.image_size, 3]),
                #                             ycrcb.reshape([n, self.image_size, self.image_size, 3]),
                #                             hsv.reshape([n, self.image_size, self.image_size, 3])],
                #                            axis=3)
                #
                #     image = np.transpose(image, (0, 3, 1, 2))
                #     image = image.astype(np.float32)
                #     image = image.reshape([n, self.channels * 3, self.image_size, self.image_size])
                #     image = image / 255.0
                #
                #     label = int(label)
                #     return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))
                #
                # elif self.mode == 'test':
                ycrcb = cv2.cvtColor(depth, cv2.COLOR_BGR2YCR_CB)
                hsv = cv2.cvtColor(ir, cv2.COLOR_BGR2HSV)

                color = color_augumentor(color, target_shape=(48, 48,3), is_infer=True)
                ycrcb = color_augumentor(ycrcb, target_shape=(48, 48,3), is_infer=True)
                hsv = color_augumentor(hsv, target_shape=(48, 48,3), is_infer=True)
                n = len(color)

                color = np.concatenate(color, axis=0)
                ycrcb = np.concatenate(ycrcb, axis=0)
                hsv = np.concatenate(hsv, axis=0)

                image = np.concatenate([color.reshape([n,48,48, 3]),
                                        ycrcb.reshape([n,48,48, 3]),
                                        hsv.reshape([n,48,48, 3])],
                                       axis=2)

                image = np.transpose(image, (0, 3, 1, 2))
                image = image.astype(np.float32)
                image = image.reshape([n, 3 * 3, 48,48])
                image = image / 255.0


                out = infer(net,torch.FloatTensor(image) )
                print('probabilities ' + str(out))
                print('done mean' + str(mean(out)))
        # else:
        #     continue
    # submission(out, save_dir + '_noTTA.txt', mode='test')


def main(config):
    if config.mode == 'train':
        run_train(config)

    if config.mode == 'infer_test':
        config.pretrained_model = r'global_min_acer_model.pth'
        run_test(config, dir='global_test_36_TTA')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default=-1)
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_mode', type=str, default='fusion')

    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--cycle_num', type=int, default=2)
    parser.add_argument('--cycle_inter', type=int, default=2)

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    config = parser.parse_args()
    print(config)
    main(config)
