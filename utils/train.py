import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models

import numpy as np
from sklearn.metrics import f1_score


def f1_score_(pred, target, threshold=0.3):
    pred = np.array(pred.cpu() > threshold, dtype=float)
    return f1_score(target.cpu(), pred, average='micro')


def select_model(model, num_classes):
    if model == 'resnet18':
        model_ = models.resnet18(pretrained=True)
        model_.fc = nn.Linear(512, num_classes)
    elif model == 'resnet34':
        model_ = models.resnet34(pretrained=True)
        model_.fc = nn.Linear(512, num_classes)
    if model == 'resnet50':
        model_ = models.resnet50(pretrained=True)
        model_.fc = nn.Linear(2048, num_classes)
    if model == 'resnet101':
        model_ = models.resnet101(pretrained=True)
        model_.fc = nn.Linear(2048, num_classes)
    return model_


class Baseline():
    def __init__(self, model, num_classes, gpu_id=0, print_freq=10, epoch_print=10, epoch_save=50):
        self.gpu = gpu_id
        self.print_freq = print_freq
        self.epoch_print = epoch_print
        self.epoch_save = epoch_save
        
        torch.cuda.set_device(self.gpu)
        
        self.loss_function = nn.BCELoss().cuda(self.gpu)
        
        model = select_model(model, num_classes)
        self.model = model.cuda(self.gpu)
        
        self.train_losses = []
        self.train_f1 = []
        self.test_losses = []
        self.test_f1 = []
        
        
    def train(self, train_data, test_data, resume=False, save=False, start_epoch=0, epochs=100, 
              lr=0.1, weight_decay=0.0001, milestones=False):
        # Model to Train Mode
        self.model.train()
        
        # Set Optimizer and Scheduler
        optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        if milestones:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [epochs//2, epochs*3//4], gamma=0.1)
        
        # Optionally Resume from Checkpoint
        if resume:
            if os.path.isfile(resume):
                print('=> Load checkpoint from {}'.format(resume))
                loc = 'cuda:{}'.format(self.gpu)
                checkpoint = torch.load(resume, map_location=loc)
                
                self.model.load_state_dict(checkpoint['state_dict'])
                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                print('=> Loaded checkpoint from {} with epoch {}'.format(resume, checkpoint['epoch']))
            else:
                print('=> No checkpoint found at {}'.format(resume))
        
        # Train
        for epoch in range(start_epoch, epochs):
            if epoch % self.epoch_print == 0:
                print('Epoch {} Started...'.format(epoch+1))
            for i, (X, y) in enumerate(train_data):
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                y = y.double()
                output = torch.sigmoid(self.model(X))
                output = output.double()
                loss = self.loss_function(output, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (i+1) % self.print_freq == 0:
                    train_f1 = f1_score_(output, y)
                    test_f1, test_loss = self.test(test_data)
                    
                    self.train_losses.append(loss.item())
                    self.train_f1.append(train_f1)
                    self.test_losses.append(test_loss)
                    self.test_f1.append(test_acc)
                    
                    self.model.train()
                    
                    if epoch % self.epoch_print == 0:
                        print('Iteration : {} - Train Loss : {:.2f}, Test Loss : {:.2f}, '
                              'Train F1 : {:.2f}, Test F1 : {:.2f}'.format(i+1, loss.item(), test_loss, train_f1, test_f1))
                    
            scheduler.step()
            if save and (epoch % self.epoch_save == 0):
                save_checkpoint(self.depth, self.num_classes, self.pretrained, epoch,
                                state={'epoch': epoch+1, 'state_dict':self.model.state_dict(),
                                       'optimizer':optimizer.state_dict(), 'scheduler':scheduler})
            
            
    def test(self, test_data):
        f1, total = 0, 0
        losses = []
        
        # Model to Eval Mode
        self.model.eval()
        
        # Test
        with torch.no_grad():
            for i, (X, y) in enumerate(test_data):
                X, y = X.cuda(self.gpu, non_blocking=True), y.cuda(self.gpu, non_blocking=True)
                y = y.double()
                output = torch.sigmoid(self.model(X))
                output = output.double()

                loss = self.loss_function(output, y)
                losses.append(loss.item())

                f1 += f1_score_(output, y)
                total += y.size(0)
        print(len(losses))
        print(total)
        return (f1/total, sum(losses)/len(losses))
