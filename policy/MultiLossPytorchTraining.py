import operator as op
from functools import reduce
from GPUtil import showUtilization as gpu_usage
import numpy as np
import torch
from odorSimulation.model.TimeCount import TimeCount
from odorSimulation.model.log_utils import Logger
from tqdm import tqdm
from policy.PytorchTraining import PytorchRunner

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def getImgMask(crop_range, img_shape):
    # crop_range = (bsize, 2, 2)
    # image_shape = (bsize, 2, 2)
    img_top_left = crop_range[..., 0]
    img_bot_right = crop_range[..., 1]

    masks = []
    for i, img_size in enumerate(img_shape):
        values = torch.arange(0, img_size).view(
            (1,) * len(crop_range.shape[:-2]) + (img_size,)).to(crop_range.device)
        mask = (img_top_left[..., i, None] <= values) * (values < img_bot_right[..., i, None])
        mask = mask.unsqueeze(-1 - i)
        masks.append(mask)
    mask = reduce(op.and_, masks)
    del values
    return mask


class MultiLossPytorchRunner(PytorchRunner):
    def __init__(self, runner_arg={}, num_epochs=100):
        super(MultiLossPytorchRunner, self).__init__(runner_arg)

    def WassersteinDistLoss(self, predict, y):
        def calculateLoss():
            # predict (bsize, time_len, 10, 30), source_distance_map = (bsize, time_len, 10, 30), source_bool = (bsize,time_len, 2)

            # loss_true_source
            loss_true_source = (normalized_predict * source_distance_map).sum((-1, -2))  # (bsize, time_len)

            # loss = torch.where(source_inside, loss_true_source, loss_false_source).mean()
            # print(loss_true_source.shape, self.X_input_valid.shape)

            self.loss_full = loss_true_source.clone().detach().to('cpu')
            loss = torch.sum(loss_true_source * self.X_input_valid) / torch.sum(self.X_input_valid)
            self.conditional_loss = loss.clone().detach().to('cpu')

            return loss

        def calculateAccuracy():
            pad_range = self.accuracy_range
            npad = ((0, 0), (0, 0), (pad_range, pad_range), (pad_range, pad_range))
            predict_padded = torch.from_numpy(
                np.pad(predict.detach().to("cpu").numpy(), npad, mode="constant", constant_values=0)).to(DEVICE)

            # clamp the positions of sources which are out of heat map
            abs_source_pos = torch.clamp(source_map_pos_index, 0, predict.shape[-1] - 1) + pad_range
            source_pos_crop = torch.stack([abs_source_pos - pad_range, abs_source_pos + 1 + pad_range], dim=-1)
            source_pos_mask = getImgMask(source_pos_crop, np.array(
                source_distance_map.shape[-2:]) + pad_range * 2)  # bsize, time_len, k_tap, k_tap

            predicted_at_source = predict_padded[source_pos_mask].reshape(bsize, time_len, pad_range * 2 + 1,
                                                                          pad_range * 2 + 1)
            predicted_at_source = torch.sum(predicted_at_source, dim=(-1, -2))  # bsize, time_len

            conditional_confident = self.X_input_valid * predicted_at_source  # normalized_predict[source_pos_mask].reshape(bsize, time_len)#[source_inside]
            confident = self.X_full_input_valid * predicted_at_source

            conditional_accuracy = (torch.sum(conditional_confident) / torch.sum(self.X_input_valid)).to('cpu')
            accuracy = (torch.sum(confident) / torch.sum(self.X_full_input_valid)).to('cpu')
            half_last_accuracy = (torch.sum(confident[:, time_len // 2:]) / torch.sum(
                self.X_full_input_valid[:, time_len // 2:])).to('cpu')

            # TODO remove
            self.accuracy_full = predicted_at_source.to('cpu')
            self.conditional_accuracy = conditional_accuracy.to('cpu')
            self.accuracy = accuracy.to('cpu')
            self.half_last_accuracy = half_last_accuracy
            return accuracy

        source_distance_map, source_pos, source_map_pos_index = y
        source_distance_map, source_pos, source_map_pos_index = source_distance_map.to(DEVICE), source_pos.to(
            DEVICE), source_map_pos_index.to(DEVICE)
        map_x, map_y = predict.shape[:-2]

        bsize, time_len = predict.shape[:2]

        predict_sum = predict.sum((-1, -2))  # + (predict.sum((-1, -2)) == 0)
        if torch.sum(predict_sum == 0) > 0:
            print("Asd")

        # normalized_predict = predict / predict_sum[..., None, None]  # (bsize, time_len, 10, 30)
        normalized_predict = predict.to(DEVICE)

        loss = calculateLoss()
        accuracy = calculateAccuracy()

        return loss, accuracy

    def dataInitialize(self, multi_datas):
        self.train_data, self.test_data = [], []
        self.X_input_valid_final_train, self.X_full_input_valid_final_train = [], []
        self.X_input_valid_final_test, self.X_full_input_valid_final_test = [], []
        for datas in multi_datas:
            data_size = datas[0].shape[0]
            train_size = int(data_size * 0.8)
            test_size = data_size - self.train_size
            train_data = [data[:self.train_size] for data in datas]
            test_data = [data[self.train_size:] for data in datas]

            self.train_data.append(train_data)  # 3, datas_size
            self.test_data.append(test_data)

            X_input_valid_final_train, X_full_input_valid_final_train = self.getXInputValid(train_data[0])
            X_input_valid_final_test, X_full_input_valid_final_test = self.getXInputValid(test_data[0])

            self.X_input_valid_final_train.append(X_input_valid_final_train)
            self.X_full_input_valid_final_train.append(X_full_input_valid_final_train)
            self.X_input_valid_final_test.append(X_input_valid_final_test)
            self.X_full_input_valid_final_test.append(X_full_input_valid_final_test)

    def logDataInitialize(self, log_datas):
        self.log_datas = log_datas
        self.source_concentration, self.water_flow_force = log_datas

    def learn(self, num_epochs=None, max_patience=3, bsize=None, shown=True):
        if bsize != None:
            self.bsize = bsize
        self.timeCount.reset()
        if num_epochs is None:
            num_epochs = self.num_epochs
        # self.modelInitialize(train_data)

        self.best_test_loss = np.inf
        self.best_model_weights = {}
        patience = 0
        # range_epoch = tqdm(range(num_epochs))
        range_epoch = range(num_epochs)
        up_range = self.model.up
        for epoch in range(num_epochs):

            self.valid = self.X_input_valid_final_test
            self.full_valid = self.X_full_input_valid_final_test
            self.runEp(self.test_data, log=self.test_log)

            print(self.test_log.getCurrent('conditional_loss_0'), self.test_log.getCurrent('conditional_accuracy_0'),  # 5
            self.test_log.getCurrent('conditional_loss_1'), self.test_log.getCurrent('conditional_accuracy_1'),  # 10
            self.test_log.getCurrent('conditional_loss_22'), self.test_log.getCurrent('conditional_accuracy_2'),  # 20
                  )

            self.valid = self.X_input_valid_final_train
            self.full_valid = self.X_full_input_valid_final_train
            self.runEp(self.train_data, log=self.train_log, update=True, shown=shown)

            # early terminate
            test_loss = self.test_log.getCurrent('conditional_loss')
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                self.best_model_weights = self.model.state_dict()
                patience = 0
            else:
                patience += 1
            if patience >= max_patience:
                print(epoch)
                self.train_log.addListItem(self.extra_dic_item[1])
                self.test_log.addListItem(self.extra_dic_item[0])
                self.saveModel()
                break
            # print(epoch, patience)
            if epoch == num_epochs - 1:
                self.saveModel()

    def predict(self, obs):
        return self.model.forward(obs)

    def getXInputValid(self, X_input_final):

        len_data, time_len = X_input_final.shape[:2]

        T_range = self.model.T_range
        T_threshold = self.model.T_threshold
        X_input_positive_cum_sum = torch.cumsum(X_input_final[..., 0] > 0, dim=-1)  # bsize. time_len
        X_full_input_valid = X_input_positive_cum_sum > 0
        X_full_input_valid = torch.ones(X_full_input_valid.shape)
        if T_threshold != 0:
            X_input_positive_cum_sum[:, T_range:] = X_input_positive_cum_sum[:, T_range:] - X_input_positive_cum_sum[:,
                                                                                            :-T_range]
            X_input_valid = (X_input_positive_cum_sum - T_threshold) >= 0

        else:
            X_input_valid = X_input_positive_cum_sum > 0

        print(torch.sum(X_input_valid) / np.product(X_input_valid.numpy().shape),
              torch.sum(X_full_input_valid) / np.product(X_full_input_valid.numpy().shape))

        return X_input_valid, X_full_input_valid  # (bsize, time_len, 1)

    def runEp(self, multiDatas, log=None, update=False, shown=False):
        total = 0
        correct = 0

        # loss_full, accuracy_full = [], []
        bsize = self.bsize

        # set iterator
        len_data = datas[0].shape[0]

        range_i = range(len_data // bsize)
        if shown: range_i = tqdm(range_i)
        shuffle_idx = np.array(range(len_data))
        np.random.shuffle(shuffle_idx)
        shuffle_idx = torch.from_numpy(shuffle_idx).type(torch.long)

        up_range = self.model.up

        conditional_loss_array, conditional_accuracy_array = [[] for i in range(up_range + 1)], [[] for i in
                                                                                                 range(up_range + 1)]
        accuracy_array, half_last_accuracy_array = [[] for i in range(up_range + 1)], [[] for i in range(up_range + 1)]

        for i in range_i:
            sample_idx = shuffle_idx[i * bsize:(i + 1) * bsize]

            for up in range(up_range + 1):
                # saved array

                datas = multi_data[up]

                sample_data = [data[sample_idx] for data in datas]
                X_input, agent_action, distance_map, source_map_pos, source_map_pos_index, = sample_data

                self.X_input_valid = self.valid[sample_idx][up].to(DEVICE)
                self.X_full_input_valid = self.full_valid[sample_idx].to(DEVICE)

                # gpu_usage()
                outputs = self.model(X_input, agent_action.type(torch.long), get_all=True)

                loss, accuracy = self.criterion(outputs, (distance_map, source_map_pos, source_map_pos_index))
                if up == 0:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss

                conditional_loss, conditional_accuracy, accuracy, half_last_accuracy = self.conditional_loss.item(), self.conditional_accuracy.item(), self.accuracy.item(), self.half_last_accuracy.item()
                conditional_loss_array[up].append(conditional_loss)
                conditional_accuracy_array[up].append(conditional_accuracy)
                accuracy_array[up].append(accuracy)
                half_last_accuracy_array[up].append(half_last_accuracy)

                loss_full[up].append(self.loss_full.clone())
                accuracy_full[up].append(self.accuracy_full.clone())

            if update:
                self.update(total_loss)

            if shown:
                # try:
                range_i.set_description(
                    "train: 5: c_loss = {:.3f}, c_acc = {:.3f}, 10: c_loss = {:.3f}, c_acc = {:.3f}, 20: c_loss = {:.3f}, c_acc = {:.3f}".format(
                        conditional_loss_array[0][-1], conditional_accuracy[0][-1], #5
                        conditional_loss_array[1][-1],conditional_accuracy[1][-1], #10
                        conditional_loss_array[2][-1], conditional_accuracy[2][-1], #20

                    ))

        dic_item = {'time_count': self.timeCount.count(),
                    }

        for up in range(up_range+1):
            dic_item['conditional_loss_{}'.format(up)] = np.mean(conditional_loss_array[up])
            dic_item['conditional_accuracy_{}'.format(up)] = np.mean(conditional_accuracy_array[up])


        self.extra_dic_item[update] = {'source_concentration': self.source_concentration[shuffle_idx].tolist(),
                                       'water_flow_force': self.water_flow_force[shuffle_idx].tolist(),
                                       'loss_full': torch.cat(loss_full, dim=0).tolist(),
                                       'accuracy_full': torch.cat(accuracy_full, dim=0).tolist(),
                                       'shuffle_idx': shuffle_idx.tolist(),
                                       }
        log.addListItem(dic_item)

    def saveModel(self):

        # process name
        folder_path = "../output/log/"
        folder_path = self.folder_path
        train_log_path = folder_path + "log/train_log/{}_{:3f}.txt".format(self.name, self.best_test_loss)
        test_log_path = folder_path + "log/test_log/{}_{:3f}.txt".format(self.name, self.best_test_loss)
        model_path = folder_path + "modelWeight/{}_{:3f}.pth".format(self.name, self.best_test_loss)

        # save log
        self.train_log.writeToFile(train_log_path)
        self.test_log.writeToFile(test_log_path)

        print(train_log_path, "\n")

        # save model
        torch.save({
            'model': self.best_model_weights,
        }, model_path)
