import torch

from utils import *
from process.augmentation import *
from process.data_helper import *
from LbpExtraction import calc_lbp


class FDDataset(Dataset):
    def __init__(self, mode, modality='color', fold_index='<NIL>', image_size=128, augment=None, balance=True):
        super(FDDataset, self).__init__()
        print('fold: ' + str(fold_index))
        print(modality)

        self.augment = augment
        self.mode = mode
        self.modality = modality
        self.balance = balance

        self.channels = 3
        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR
        self.image_size = image_size
        self.fold_index = fold_index

        self.set_mode(self.mode, self.fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        print(mode)
        print('fold index set: ', fold_index)

        if self.mode == 'test':
            self.test_list = load_test_list()
            self.num_data = len(self.test_list)
            print('set dataset mode: test')

        elif self.mode == 'val':
            self.val_list = load_val_list()
            self.num_data = len(self.val_list)
            print('set dataset mode: val')

        elif self.mode == 'train':
            self.train_list = load_train_list()
            random.shuffle(self.train_list)
            self.num_data = len(self.train_list)
            print('set dataset mode: train')

            if self.balance:
                self.train_list = transform_balance(self.train_list)

        print(self.num_data)

    def __getitem__(self, index):

        if self.fold_index is None:
            print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            return

        if self.mode == 'train':
            if self.balance:
                if random.randint(0, 1) == 0:
                    tmp_list = self.train_list[0]
                else:
                    tmp_list = self.train_list[1]

                pos = random.randint(0, len(tmp_list) - 1)
                color, depth, ir, label = tmp_list[pos]
            else:
                color, depth, ir, label = self.train_list[index]

        elif self.mode == 'val':
            color, depth, ir, label = self.val_list[index]

        elif self.mode == 'test':
            color, depth, ir = self.test_list[index]
            test_id = color + ' ' + depth + ' ' + ir

        color = cv2.imread(os.path.join(DATA_ROOT, color), 1)
        depth = color.copy()
        ir = color.copy()

        color = cv2.resize(color, (RESIZE_SIZE, RESIZE_SIZE))
        depth = cv2.resize(depth, (RESIZE_SIZE, RESIZE_SIZE))
        ir = cv2.resize(ir, (RESIZE_SIZE, RESIZE_SIZE))

        if self.mode == 'train':
            print('color shape before augmentor ' + str(color.shape))
            color = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ycrcb = cv2.cvtColor(depth, cv2.COLOR_BGR2YCR_CB)
            ycrcblbp = calc_lbp(ycrcb[:, :, 1])
            lbp = calc_lbp(cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY))

            color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3))
            ycrcblbp = color_augumentor(ycrcblbp, target_shape=(self.image_size, self.image_size, 3))
            lbp = color_augumentor(lbp, target_shape=(self.image_size, self.image_size, 3))
            print('color shape ' + str(color.shape))

            # color = cv2.resize(color, (self.image_size, self.image_size))
            # ycrcblbp = cv2.resize(ycrcblbp, (self.image_size, self.image_size))
            # lbp = cv2.resize(lbp, (self.image_size, self.image_size))

            image = np.concatenate([color.reshape([self.image_size, self.image_size, 3]),
                                    ycrcblbp.reshape([self.image_size, self.image_size, 3]),
                                    lbp.reshape([self.image_size, self.image_size, 3])],
                                   axis=2)

            if random.randint(0, 1) == 0:
                random_pos = random.randint(0, 2)
                if random.randint(0, 1) == 0:
                    image[:, :, 3 * random_pos:3 * (random_pos + 1)] = 0
                else:
                    for i in range(3):
                        if i != random_pos:
                            image[:, :, 3 * i:3 * (i + 1)] = 0

            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0

            label = int(label)
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))

        elif self.mode == 'val':
            color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            depth = color_augumentor(depth, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            ir = color_augumentor(ir, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(color)

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ycrcb = cv2.cvtColor(depth, cv2.COLOR_BGR2YCR_CB)
            lbp = calc_lbp(ir)
            gray = cv2.resize(gray, (self.image_size, self.image_size))
            ycrcb = cv2.resize(ycrcb, (self.image_size, self.image_size))
            lbp = cv2.resize(lbp, (self.image_size, self.image_size))

            image = np.concatenate([gray.reshape([self.image_size, self.image_size, 3]),
                                    ycrcb.reshape([self.image_size, self.image_size, 3]),
                                    lbp.reshape([self.image_size, self.image_size, 3])],
                                   axis=2)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0

            label = int(label)
            return torch.FloatTensor(image), torch.LongTensor(np.asarray(label).reshape([-1]))


        elif self.mode == 'test':
            color = color_augumentor(color, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            depth = color_augumentor(depth, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            ir = color_augumentor(ir, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(color)

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ycrcb = cv2.cvtColor(depth, cv2.COLOR_BGR2YCR_CB)
            lbp = calc_lbp(ir)
            gray = cv2.resize(gray, (self.image_size, self.image_size))
            ycrcb = cv2.resize(ycrcb, (self.image_size, self.image_size))
            lbp = cv2.resize(lbp, (self.image_size, self.image_size))

            image = np.concatenate([gray.reshape([self.image_size, self.image_size, 3]),
                                    ycrcb.reshape([self.image_size, self.image_size, 3]),
                                    lbp.reshape([self.image_size, self.image_size, 3])],
                                   axis=2)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels * 3, self.image_size, self.image_size])
            image = image / 255.0
            return torch.FloatTensor(image), test_id

    def __len__(self):
        return self.num_data


# check #################################################################
def run_check_train_data():
    dataset = FDDataset(mode='train')
    print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label = dataset[m]
        print(image.shape)
        print(label)

        if m > 100:
            break


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()
