"""
INSERT YOUR NAME HERE
Woo Hyun Maeng
"""

from __future__ import print_function
from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#from tensorboardX import SummaryWriter  # for pytorch below 1.14
from torch.utils.tensorboard import SummaryWriter # for pytorch above or equal 1.14


# Cifar 10 is 32 x 32 x image
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ## 3 input image channel, 32 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        ## 32 input image channel, 32 output channels, 3x3 square convolution
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        ## 32 input image channel, 64 output channels, 3x3 square convolution
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        ## 64 input image channel, 64 output channels, 3x3 square convolution
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)

        ## maxpool 2x2; make height and width half
        self.pool = nn.MaxPool2d(2, 2)

        ## input = channels * height * width, output = 512
        ###self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc1 = nn.Linear(128 * 4 * 4, 2048)

        ## batch normalization layer
        ###self.batnorm1 = nn.BatchNorm1d(512)
        self.batnorm1 = nn.BatchNorm1d(2048)

        ## add 1 more fully connected layer at the second last of the end.
        ###self.fc3 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(2048, 512)


        ## input = 512, output = 10 classes
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.batnorm1(x)
        x = F.relu(x)
        x = F.relu(self.fc3(x))
        x = self.fc2(x)

        """
        x = x.view(-1, self.num_flat_features(x))

        # first fully connected layer
        x = self.fc1(x)
        # batchnorm layer added into forward
        x = self.batnorm1(x)

        x = F.relu(x)

        ## added fully connected layer at the second last layer. named fc3.
        x = F.relu(self.fc3(x))

        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        """
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def eval_net(dataloader):
    correct = 0
    total = 0
    total_loss = 0
    net.eval() # Why would I do this?
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for data in dataloader:
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
    net.train() # Why would I do this?
    return total_loss / total, correct.float() / total


if __name__ == "__main__":
    BATCH_SIZE = 32 #mini_batch size
    MAX_EPOCH = 10 #maximum epoch to train

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    print('Building model...')
    net = Net().cuda()

    """
    parametors = net.state_dict()
    pretrained_parametors = torch.load('mytraining_with_batchnorm.pth')
    for key, _ in parametors.items():
        if key != 'fc2.weight' and key != 'fc2.bias' and key != 'fc3.weight' and key != 'fc3.bias' and key != 'conv5.weight' and key != 'conv5.bias' and key != 'conv6.weight' and key != 'conv6.bias':
            parametors[key] = pretrained_parametors[key]
    net.load_state_dict(parametors)
    """

    net.train() # Why would I do this?

    writer = SummaryWriter(log_dir='./log')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    print('Start training...')
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(MAX_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # freeze batchnormalization layer
            #for param in net.parameters():
            #    if param == 'batnorm1':
            #        param.requires_grad = False


            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 2000 mini-batches
                print('    Step: %5d avg_batch_loss: %.5f' %
                      (i + 1, running_loss / 500))
                running_loss = 0.0
        print('    Finish training this EPOCH, start evaluating...')
        train_loss, train_acc = eval_net(trainloader)
        test_loss, test_acc = eval_net(testloader)
        print('EPOCH: %d train_loss: %.5f train_acc: %.5f test_loss: %.5f test_acc %.5f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))

        #writer.add_scalar('train_loss', train_loss, epoch)
        #writer.add_scalar('test_loss', test_loss, epoch)
        #writer.add_scalars(main_tag='Adam Loss', tag_scalar_dict={'train_loss': train_loss, 'test_loss': test_loss}, global_step=epoch)
        #writer.add_scalars(main_tag='Adam Accuracy', tag_scalar_dict={'train_accuracy': train_acc, 'test_accuracy': test_acc}, global_step=epoch)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
    for i in range(10):
        writer.add_scalars(main_tag='moreconv Loss', tag_scalar_dict={'train_loss': train_loss_list[i], 'test_loss': test_loss_list[i]}, global_step=i)
        writer.add_scalars(main_tag='moreconv Accuracy', tag_scalar_dict={'train_accuracy': train_acc_list[i], 'test_accuracy': test_acc_list[i]}, global_step=i)
    # writer.close()

    writer.close()
    print('Finished Training')
    print('Saving model...')
    torch.save(net.state_dict(), 'q4_model.pth')
