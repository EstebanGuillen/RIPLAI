import fastai
from fastai.vision import *

img_size=224
path = '/opt/AIStorage/PLAYGROUND/images/' + str(img_size) + '/'
data = ImageDataBunch.from_folder(path,valid='validation', size=img_size, bs=16, ds_tfms=(rand_pad(2, img_size), []))
epochs=12

arch_list = [('resnet18',models.resnet18), 
             ('resnet34',models.resnet34) , 
             ('resnet50',models.resnet50), 
             ('resnet101',models.resnet101), 
             ('densenet121',models.densenet121),
             ('densenet169',models.densenet169), 
             ('densenet201',models.densenet201),
             ('squeezenet1_0',models.squeezenet1_0),
             ('squeezenet1_1',models.squeezenet1_1),
             ('alexnet',models.alexnet),
             ('vgg16_bn',models.vgg16_bn),
             ('vgg19_bn',models.vgg19_bn)]

for name, arch in arch_list:
    print(name)
    learn = cnn_learner(data, arch, metrics=accuracy, bn_final=True).to_fp16()
    #learn = cnn_learner(data, arch, metrics=accuracy, bn_final=True) 
    learn.model = nn.DataParallel(learn.model)
    learn.fit_one_cycle(epochs)
