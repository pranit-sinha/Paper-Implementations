with open('network.nnet') as f:
    line = f.readline()
    cnt = 1
    while line[0:2] == "//":
        line = f.readline()
        cnt += 1
    numLayers, inputSize, outputSize, maxLayersize = [int(x) for x in line.strip().split(",")[:-1]] # does't include the input layer!
    line = f.readline()

    layerSizes = [int(x) for x in line.strip().split(",")[:-1]] # input layer size, layer1size, layer2size...

    line = f.readline()
    layerSizes = [int(x) for x in line.strip().split(",")[:-1]]
    print(layerSizes)
