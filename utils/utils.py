import numpy as np
import torch


def noise(size, cuda=False):
    noise = torch.from_numpy(np.random.normal(0.0, size=size)).float()
    if cuda:
        noise = noise.cuda()
    return noise

def only_good_classes(brain_data, brain_data_tags):
    
    unique_brain_data_tags = np.unique(brain_data_tags, axis=0)
    goodClasses = set()
    totalGoodSamples = 0

    for i in range(len(unique_brain_data_tags)):
        class_data = brain_data[np.argmax(brain_data_tags, axis=1) == np.argmax(unique_brain_data_tags[i])]
        if len(class_data) > 30:
            print(i, class_data.shape)
            goodClasses.add(np.argmax(unique_brain_data_tags[i]))
            totalGoodSamples += len(class_data)
            print("total good samples:", totalGoodSamples)

    print(goodClasses)
    

    brain_data_good = np.zeros((totalGoodSamples,) + brain_data[0].shape)
    brain_data_tags_good = np.zeros((totalGoodSamples,) + brain_data_tags[0].shape)
    print(brain_data_good.shape, brain_data_tags_good.shape)

    curId = 0
    for i in range(len(brain_data)):
        if not np.argmax(brain_data_tags[i]) in goodClasses:
            continue
        brain_data_good[curId] = brain_data[i]
        brain_data_tags_good[curId] = brain_data_tags[i]
        curId += 1
    
    return brain_data_good, brain_data_tags_good