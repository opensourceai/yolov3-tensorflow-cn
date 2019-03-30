
for txt in ["labels.txt", "test.txt", "train.txt"]:
    new_txt = open("new_" + txt, "w")
    for line in open(txt).readlines():
        new_line = line.replace("./raccoon_dataset/", "../data/train_dome_data/")
        new_txt.write(new_line)
