import copy
import time

import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics

import p3graph

log = p3graph.utils.get_logger()


class Coach:

    def __init__(self, trainset, devset, model, opt, args):
        self.trainset = trainset
        self.devset = devset
        self.model = model
        self.opt = opt
        self.args = args
        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt.optimizer, step_size=80, gamma=0.7)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.opt.optimizer,
        #     mode='max',  # 因为你监控的是F1分数，希望它越大越好
        #     factor=0.8,  # 学习率衰减因子
        #     patience=10, # 容忍10轮性能不提升
        # )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.opt.optimizer,
        #     T_max=200,
        #     eta_min=self.args.learning_rate / 20  # 最小学习率为初始值的十分之一
        # )

        self.params = [p for name, p in self.model.named_parameters() if not name.startswith("pub_disc.")]
        self.d_params = list(self.model.pub_disc.parameters())

        self.disc_optimizer = torch.optim.Adam(self.d_params, lr=self.args.learning_rate)
        self.disc_scheduler = torch.optim.lr_scheduler.StepLR(self.disc_optimizer, step_size=80, gamma=0.7)
        # self.disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.disc_optimizer,
        #     mode='max',
        #     factor=0.8,
        #     patience=10,
        #     verbose=True
        # )
        # self.disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.disc_optimizer,
        #     T_max=200,
        #     eta_min=self.args.learning_rate / 20
        # )

    def load_ckpt(self, ckpt):
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)

    def train(self):
        log.debug(self.model)
        best_dev_f1, best_epoch, best_state = self.best_dev_f1, self.best_epoch, self.best_state

        visualizer = p3graph.TrainingVisualizer()
        visualizer.start()

        try:
            for epoch in range(1, self.args.epochs + 1):
                # if epoch <= 100:
                #     self.model.weight_adv = 0.0
                # else:
                #     self.model.weight_adv = 0.01
                epoch_loss, epoch_d_loss, epoch_disc_acc = self.train_epoch(epoch)
                visualizer.add_loss(epoch, epoch_loss)
                visualizer.add_disc_loss(epoch, epoch_d_loss)
                visualizer.add_disc_acc(epoch, epoch_disc_acc)

                dev_f1, dev_acc = self.evaluate()
                visualizer.add_f1(epoch, dev_f1)

                log.info("[Dev set] [f1 {:.4f}] [acc {:.4f}]".format(dev_f1, dev_acc))
                if best_dev_f1 is None or dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    best_epoch = epoch
                    best_state = copy.deepcopy(self.model.state_dict())
                    log.info("Save the best model.")

                # if dev_f1 > 0.68:
                #     log.info(f"早停：验证集F1分数 {dev_f1:.4f} > 0.68，在第{epoch}轮停止训练")
                #     break

        except KeyboardInterrupt:
            log.info("Training interrupted by user")
        finally:
            visualizer.stop()

        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1, dev_acc = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}] [acc {:.4f}]".format(dev_f1, dev_acc))

        return best_dev_f1, best_epoch, best_state

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_total_loss = 0
        epoch_d_train_loss = 0

        self.model.train()
        self.model.pub_disc.reset_metrics()

        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            data = self.trainset[idx]
            for k, v in data.items():
                data[k] = v.to(self.args.device)

            # --- 1. Train Discriminator (D) ---
            # if epoch > 80 and idx % 1 == 0:
            if idx % 2 == 0:
                for param in self.params:
                    param.requires_grad = False
                for param in self.d_params:
                    param.requires_grad = True

                self.disc_optimizer.zero_grad()
                _, disc_loss_d_step = self.model.get_loss(data, mode='disc_only')
                disc_loss_d_step.backward()
                torch.nn.utils.clip_grad_norm_(self.d_params, max_norm=1.0)  # Clip D's gradients
                self.disc_optimizer.step()

                epoch_d_train_loss += disc_loss_d_step.item()

            # --- 2. Train Main ---
            for param in self.params:
                param.requires_grad = True
            for param in self.d_params:
                param.requires_grad = False

            self.opt.optimizer.zero_grad()
            total_loss_step, _ = self.model.get_loss(data, mode='full')
            total_loss_step.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(self.params,max_norm=self.args.clip if hasattr(self.args, 'clip') else 1.0)
            self.opt.step()

            epoch_total_loss += total_loss_step.item()

        for param in self.model.parameters():
            param.requires_grad = True

        self.scheduler.step()
        self.disc_scheduler.step()

        end_time = time.time()

        disc_acc = self.model.pub_disc.get_accuracy()
        current_lr = self.scheduler.get_last_lr()[0]

        avg_loss = epoch_total_loss / len(self.trainset) if len(self.trainset) > 0 else 0
        avg_d_loss = epoch_d_train_loss / len(self.trainset) if len(self.trainset) > 0 else 0

        log.info(f"[Epoch {epoch}] "
                 f"[LR: {current_lr:.6f}] "
                 f"[Loss: {avg_loss:.4f}] "
                 f"[D Loss: {avg_d_loss:.4f}] "
                 f"[D Acc: {disc_acc:.4f}] "
                 f"[Time: {end_time - start_time:.2f}s]")

        return avg_loss, avg_d_loss, disc_acc

    def evaluate(self):
        dataset = self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="dev eval"):
                data = dataset[idx]
                golds.append(data["label_tensor"].cpu())
                for k, v in data.items():
                    data[k] = v.to(self.args.device)
                y_hat = self.model(data)  # model.forward()
                preds.append(y_hat.detach().to("cpu"))
            golds = torch.cat(golds, dim=0).numpy()
            preds = torch.cat(preds, dim=0).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted", zero_division=0)
            acc = metrics.accuracy_score(golds, preds)
        return f1, acc