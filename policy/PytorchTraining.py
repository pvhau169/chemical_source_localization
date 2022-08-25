import operator as op
from functools import reduce
from GPUtil import showUtilization as gpu_usage
import numpy as np
import torch
from odorSimulation.model.TimeCount import TimeCount
from odorSimulation.model.log_utils import Logger
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PytorchRunner:
    def __init__(self, runner_arg={}, num_epochs=100):
        super(PytorchRunner, self).__init__()

        self.model = None

        # Loss and optimizer
        self.criterion = None
        self.optimizer = None
        self.learning_rate = 0.01

        # training parameters
        self.bsize = 80
        self.num_epochs = num_epochs

        # logger
        self.train_log = Logger()
        self.test_log = Logger()
        self.extra_dic_item = [{}, {}]

        # timer
        self.timeCount = TimeCount()
        self.accuracy_range = 0

        self.masks = torch.tensor([[]])
        for key, value in runner_arg.items():
            setattr(self, key, value)

    def WassersteinDistLoss(self, predict, y, evaluate=False):
        def calculateLoss():
            # predict (bsize, time_len, total_centroid),
            # distance_map = (bsize, time_len, total_centroid),

            # print(normalized_predict.shape, distance_map.shape)
            loss_true_source = (normalized_predict * distance_map).sum(-1)  # (bsize, time_len)


            # self.loss_full = loss_true_source.clone().detach().to('cpu')
            conditional_loss = loss_true_source * self.X_input_valid  # bsize, time_len
            regularizer = (normalized_predict * torch.log(normalized_predict)).sum(-1) * self.X_input_valid
            if self.new_loss:
                conditional_loss = (torch.sum(conditional_loss, dim=1) / torch.sum(self.X_input_valid, dim=1)).mean()
                regularizer = (torch.sum(regularizer, dim=1) / torch.sum(self.X_input_valid, dim=1)).mean()
            else:
                conditional_loss = torch.sum(conditional_loss) / torch.sum(self.X_input_valid)
                regularizer = torch.sum(regularizer) / torch.sum(self.X_input_valid)

            conditional_loss = conditional_loss + regularizer * self.regularizer_constant

            self.conditional_loss = conditional_loss.clone().detach().to('cpu')

            return conditional_loss  # + regularizer * 0.15

        def calculateAccuracy():

            # clamp the positions of sources which are out of heat map
            # print(type(total_centroid), type(source_region))
            # print(source_region.shape)
            source_region_mask = torch.eye(total_centroid).to(DEVICE)[
                source_region.type(torch.long)]  # bsize, time_len, total_centroid

            predicted_at_source = normalized_predict * source_region_mask.to(DEVICE)  # bsize, time_len, total_centroid
            predicted_at_source = torch.sum(predicted_at_source, dim=-1)  # bsize, time_len

            conditional_confident = self.X_input_valid * predicted_at_source  # normalized_predict[source_pos_mask].reshape(bsize, time_len)#[source_inside]
            confident = self.X_full_input_valid * predicted_at_source

            # conditional_accuracy = (torch.sum(conditional_confident) / torch.sum(self.X_input_valid)).to('cpu')
            conditional_accuracy = (conditional_confident.sum(-1) / self.X_input_valid.sum(-1)).mean().to('cpu')
            accuracy = (torch.sum(confident) / torch.sum(self.X_full_input_valid)).to('cpu')

            # TODO remove

            self.conditional_accuracy = conditional_accuracy.to('cpu')
            self.accuracy = accuracy.to('cpu')
            return accuracy

        def calculateError():
            def findAngle(A, B):
                cos_between = torch.sum(A * B, axis=-1) / (torch.norm(B, dim=-1) * torch.norm(A, dim=-1))
                error_angle = torch.arccos(cos_between)
                error_angle[error_angle > np.pi] = np.pi * 2 - error_angle[error_angle > np.pi]
                error_angle[torch.isnan(error_angle)] = -1  # bsize, time_len
                return error_angle

            mask_inside_with_detect = source_inside * source_detect
            mask_inside_no_detect = source_inside * (1 - source_detect)
            mask_outside_with_detect = (1 - source_inside) * source_detect
            mask_outside_no_detect = (1 - source_inside) * (1 - source_detect)

            masks = [mask_inside_with_detect, mask_inside_no_detect, mask_outside_with_detect, mask_outside_no_detect]
            error_distance_list, error_angle_list = [], []

            # print(self.centroids.is_cuda, normalized_predict.is_cuda)
            predict_centroid = self.centroids.to(DEVICE)[torch.argmax(normalized_predict, dim=-1)]  # bsize, time_len, 2
            # predict_centroid = agent_pos + predict_centroid  # bsize, time_len, 2

            # error distance
            error_distance = torch.nn.PairwiseDistance(p=2)(predict_centroid.reshape(-1, 2),
                                                            source_pos.reshape(-1, 2)).reshape(
                source_pos.shape[:-1])  # bsize, time_len
            error_angle = findAngle(predict_centroid, source_pos)  # bsize, time_len
            if evaluate:
                return torch.stack(masks, dim=0), error_distance, error_angle

            for mask in masks:
                distance_mask, angle_mask = mask, mask * (error_angle >= 0)

                avg_error_distance = torch.sum(error_distance * distance_mask) / (torch.sum(distance_mask) + (torch.sum(distance_mask) == 0))
                avg_error_angle = torch.sum(error_angle * angle_mask) / (torch.sum(angle_mask) + (torch.sum(distance_mask) == 0))

                error_distance_list.append(avg_error_distance.to('cpu').item())
                error_angle_list.append(avg_error_angle.to('cpu').item())

            return np.array(error_distance_list), np.array(error_angle_list)

        # predict = bsize, time_len, total_centroid
        distance_map_final, source_region, source_pos, source_inside, source_detect = y
        distance_map, source_region, source_pos, source_inside, source_detect = distance_map_final.to(
            DEVICE), source_region.to(
            DEVICE), source_pos.to(DEVICE), source_inside.to(DEVICE), source_detect.to(DEVICE)

        bsize, time_len, total_centroid = predict.shape

        normalized_predict = predict.to(DEVICE)

        loss = calculateLoss()
        accuracy = calculateAccuracy()

        if evaluate == False:
            self.error_distance_list, self.error_angle_list = calculateError()
        else:
            self.masks, self.error_distance_list, self.error_angle_list = calculateError()

        return loss, accuracy

    def modelInitialize(self, model):
        self.model = model

        # Loss and optimizer
        self.criterion = self.WassersteinDistLoss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update(self, loss):

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def dataInitialize(self, datas):
        data_size = datas[0].shape[0]
        self.train_size = int(data_size * 0.8)
        self.test_size = data_size - self.train_size

        self.train_index = np.array(range(data_size))
        # np.random.shuffle(self.train_index)
        self.train_index, self.test_index = self.train_index[:self.train_size], self.train_index[self.train_size:]

        self.train_data = [data[self.train_index] for data in datas]
        self.test_data = [data[self.test_index] for data in datas]
        self.X_input_valid_final_train, self.X_full_input_valid_final_train = self.getXInputValid(self.train_data)
        self.X_input_valid_final_test, self.X_full_input_valid_final_test = self.getXInputValid(self.test_data)

    def logDataInitialize(self, log_datas):
        self.source_concentration, self.water_flow_force, self.centroids = log_datas

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
        for epoch in range(num_epochs):
            self.valid = self.X_input_valid_final_test
            self.full_valid = self.X_full_input_valid_final_test
            self.runEp(self.test_data, log=self.test_log)

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
                # self.train_log.addListItem(self.extra_dic_item[1])
                # self.test_log.addListItem(self.extra_dic_item[0])
                self.saveModel()
                break
            # print(epoch, patience)
            if epoch == num_epochs - 1:
                self.saveModel()

    def predict(self, obs):
        return self.model.forward(obs)

    def getXInputValid(self, data):
        X_input_final, source_detect = data[0], data[-1]

        len_data, time_len = X_input_final.shape[:2]
        T_range = self.model.T_range
        T_threshold = self.model.T_threshold
        X_input_positive_cum_sum = torch.cumsum(X_input_final[..., 0] > 0, dim=-1)  # bsize. time_len
        X_full_input_valid = X_input_positive_cum_sum > 0
        X_full_input_valid = torch.ones(X_full_input_valid.shape)

        if self.new_loss:
            X_input_valid = source_detect
            X_input_valid[:, -1] = 1
        else:
            X_input_valid = torch.ones(source_detect.shape)

        print(torch.sum(X_input_valid) / np.product(X_input_valid.numpy().shape),
              torch.sum(X_full_input_valid) / np.product(X_full_input_valid.numpy().shape))

        return X_input_valid, X_full_input_valid  # (bsize, time_len, 1)

    def runEp(self, datas, log=None, update=False, shown=False, evaluate=False):
        # print("asd")
        total = 0
        correct = 0
        conditional_loss_array, conditional_accuracy_array = [], []
        accuracy_array, half_last_accuracy_array = [], []
        loss_full, accuracy_full = [], []
        error_angle_array, error_distance_array = [], []


        masks, source_distance_array,valid_array = [], [], []
        bsize = self.bsize

        # set iterator
        len_data = datas[0].shape[0]

        range_i = range(len_data // bsize)
        if shown: range_i = tqdm(range_i)
        shuffle_idx = np.array(range(len_data))
        np.random.shuffle(shuffle_idx)
        shuffle_idx = torch.from_numpy(shuffle_idx).type(torch.long)

        for i in range_i:

            sample_idx = shuffle_idx[i * bsize:(i + 1) * bsize]
            sample_data = [data[sample_idx] for data in datas]
            X_input, agent_action, distance_map_final, source_region, source_pos, source_inside, source_detect = sample_data

            self.X_input_valid = self.valid[sample_idx].to(DEVICE)
            self.X_full_input_valid = self.full_valid[sample_idx].to(DEVICE)

            outputs = self.model(X_input, agent_action.type(torch.long), get_all=True)

            loss, accuracy = self.criterion(outputs,
                                            (distance_map_final, source_region, source_pos, source_inside,
                                             source_detect), evaluate = evaluate)
            if update:
                self.update(loss)

            conditional_loss, conditional_accuracy, accuracy = self.conditional_loss.item(), self.conditional_accuracy.item(), self.accuracy.item()
            conditional_loss_array.append(conditional_loss)
            conditional_accuracy_array.append(conditional_accuracy)
            error_angle_array.append(self.error_angle_list)
            error_distance_array.append(self.error_distance_list)
            accuracy_array.append(accuracy)

            if evaluate:
                masks.append(self.masks)
                source_distance_array.append(torch.cdist(source_pos, torch.tensor([[0, 0]], dtype=torch.float)).squeeze(-1))
                valid_array.append(self.X_input_valid)

            if shown:
                range_i.set_description(
                    "train: loss= {:.2f}, acc= {:.2f}, dis_in= {:.2f}, dis_in_no_detect= {:.2f}, dis_out= {:.2f}, dis_out_no_detect= {:.2f}, ang_in= {:.2f}, ang_in_no_detect= {:.2f}, ang_out= {:.2f}, ang_out_no_detect= {:.2f} | test: loss={:.2f}, acc ={:.2f}".format(
                        conditional_loss, conditional_accuracy, *self.error_distance_list, *self.error_angle_list,
                        self.test_log.getCurrent('conditional_loss'), self.test_log.getCurrent('conditional_accuracy')))

                # range_i.set_description(
                #     "train: loss= {:.2f}, acc= {:.2f}, dis= {:.2f}, ang= {:.2f}| test: {:.2f}, acc= {:.2f}, dis= {:.2f}, ang= {:.2f}".format(
                #         conditional_loss, conditional_accuracy, self.avg_error_distance, self.avg_error_angle,
                #         self.test_log.getCurrent('conditional_loss'), self.test_log.getCurrent('conditional_accuracy'),
                #         self.test_log.getCurrent('avg_error_distance'), self.test_log.getCurrent('avg_error_angle')))

        if evaluate:
            return torch.cat(masks, dim=1), torch.cat(error_distance_array, dim=0), torch.cat(error_angle_array,dim=0), \
                   torch.cat(source_distance_array, dim=0), torch.cat(valid_array, dim =0)
        error_distance_array = np.array(error_distance_array).mean(axis=0)
        error_angle_array = np.array(error_angle_array).mean(axis=0)

        dic_item = {'time_count': self.timeCount.count(),
                    'conditional_loss': np.mean(conditional_loss_array),
                    'accuracy': np.mean(accuracy_array),
                    'conditional_accuracy': np.mean(conditional_accuracy_array),

                    'error_distance_inside': error_distance_array[0],
                    'error_distance_inside_no_detect': error_distance_array[1],
                    'error_distance_outside': error_distance_array[2],
                    'error_distance_outside_no_detect': error_distance_array[3],

                    'error_angle_inside': error_angle_array[0],
                    'error_angle_inside_no_detect': error_angle_array[1],
                    'error_angle_outside': error_angle_array[2],
                    'error_angle_outside_no_detect': error_angle_array[3],
                    }

        # self.extra_dic_item[update] = {'source_concentration': self.source_concentration[shuffle_idx].tolist(),
        #                                'water_flow_force': self.water_flow_force[shuffle_idx].tolist(),
        #                                'loss_full': torch.cat(loss_full, dim=0).tolist(),
        #                                'accuracy_full': torch.cat(accuracy_full, dim=0).tolist(),
        #                                'shuffle_idx': shuffle_idx.tolist(),
        #                                }
        log.addListItem(dic_item)

    def saveModel(self):

        # process name
        folder_path = "../output/log/"
        folder_path = self.folder_path
        train_log_path = folder_path + "log/train_log/{}.txt".format(self.name)
        test_log_path = folder_path + "log/test_log/{}.txt".format(self.name)
        model_path = folder_path + "modelWeight/{}.pth".format(self.name)

        # save log
        self.train_log.writeToFile(train_log_path)
        self.test_log.writeToFile(test_log_path)

        print(train_log_path, "\n")

        # save model
        torch.save({
            'model': self.best_model_weights,
        }, model_path)
