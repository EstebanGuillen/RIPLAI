from fastai.vision import *

#path = '/opt/AIStorage/PLAYGROUND/'


#data = ImageDataBunch.from_csv(path,folder='images',ds_tfms=(rand_pad(2, 1024), []), size=1024, bs=32)



path = '/opt/AIStorage/PLAYGROUND/images/1024/'
image_size = 512
epochs = 12
data = ImageDataBunch.from_folder(path,valid='validation', size=image_size, bs=32, ds_tfms=(rand_pad(2, image_size), []))

arch = models.densenet201
learn = cnn_learner(data, arch, metrics=accuracy)
learn.to_fp16()
learn.model = nn.DataParallel(learn.model)

#learn.fit_one_cycle(epochs, 0.003)

learn.fit(epochs, 0.001)  
