import torch
import customDatasetTest
import torchvision.models as models
import torch.nn as nn
import os
import torch.backends.cudnn as cudnn

dataset = '../../datasets/iMaterialistFashion' # Path to dataset
split = '/anns/validation'
batch_size = 500
workers = 12
model_name = 'CNN_pytorchmarginloss_resnet50_epoch_38'

if not os.path.exists('../../datasets/iMaterialistFashion/CNN_output/' + model_name):
    os.makedirs('../../datasets/iMaterialistFashion/CNN_output/' + model_name)

output_file_path = '../../datasets/iMaterialistFashion/CNN_output/' + model_name + '/' + split.split('/')[-1] + '.txt'
output_file = open(output_file_path, "w")


checkpoint = torch.load(dataset + '/models/CNN_pytorchmarginloss/' + model_name + '.pth.tar')
# model = checkpoint["model"] # I will be able to do that when I save the model structure also
# weights = model.load_state_dict(checkpoint)
# As i havent saved the model, I have to create it
model = models.__dict__['resnet50']()
num_classes = 228
del(model._modules['fc'])
model.fc = nn.Linear(2048,num_classes)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(checkpoint['state_dict'])
model.sm = nn.Sigmoid()

test_dataset = customDatasetTest.customDatasetTest(dataset, split, CenterCrop=224)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=None)

with torch.no_grad():
    for i,data in enumerate(test_loader):
        images, labels, indices = data
        outputs = model(images)
        for idx,el in enumerate(outputs):
            topic_probs_str = ''
            for t in el:
                topic_probs_str = topic_probs_str + ',' + str(float(t))
            output_file.write(indices[idx] + '.jpg' + topic_probs_str + '\n')
            print topic_probs_str
        if i % 1000 == 1: print(str(i) + ' / ' + str(len(test_loader)))
