from torch.autograd import Variable

from utils import AverageMeter

def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    
    train_loss = AverageMeter()
    
    curr_iter = (epoch - 1) * len(train_loader)
    total_loss_train = []
    for i, data in enumerate(train_loader):
        images, targets = data
        images = Variable(images).to(device)
        targets = Variable(targets).to(device)
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item())
        curr_iter += 1
        if epoch % 10 == 0:
            if (i + 1) % 8 == 0:
                print('[epoch %d], [iter %d / %d], [train loss %.5f]' % (
                    epoch, i + 1, len(train_loader), train_loss.avg))
        total_loss_train.append(train_loss.avg)
    return train_loss.avg, total_loss_train