# -*- coding: utf-8 -*-
from glob import glob
from dataPreprocess import FileOperation,TransformPytorchDataType
from cnnModel import VGGNet, ModelOperation
import torch.optim as optim
import torch.nn as nn
import random
from PIL import Image

# step1. splite data
c1_path = glob('.\Dog_Images\Chihuahua\*.jpg')
c2_path = glob('.\Dog_Images\Japanese_spaniel\*.jpg')
c3_path = glob('.\Dog_Images\Maltese_dog\*.jpg')
path_list = [c1_path,c2_path,c3_path]
FileOperation().getDatabase(path_list)

# step2. transform into datasets (pytorch)
batch_size = 16
learning_rate = 0.0002
epoch = 10
database = TransformPytorchDataType(batch_size)
train_data = database.trainDataLoader()
val_data = database.valDataLoader()

# step3. model
cnn_model = ModelOperation(VGGNet(freeze=True))
cnn_model.structure((3, 224, 224))
optimizer = optim.Adam(cnn_model.model.parameters(), lr=learning_rate)
loss_func = nn.CrossEntropyLoss()
cnn_model.compiler(optimizer, loss_func)
cnn_model.save_model('VGG16.pth')

# step4. train
cnn_model.fit(train_data,val_data,epoch)

# step5. predict
r1 = random.choice(c1_path)
r2 = random.choice(c2_path)
r3 = random.choice(c3_path)
random_list = [r1,r2,r3]
test_data = [Image.open(r) for r in random_list]
cnn_model.load_model()
result = cnn_model.predict(test_data)
class_idx, class_name = cnn_model.classifier(['Chihuahua','Japanese_spaniel','Maltese_dog'])
print(class_name)