import os

file_type = ".mp4"
split = "val"
dataset_root = "/media/data2/Kinetics/kinetics-dataset/k600/"
classes_list_path = "class_list.txt"
val_csv_out_path = "val.csv"
separator = ","

with open(classes_list_path, "r") as f:
    lines = [line.rstrip() for line in f.readlines()]

ordered_classes = sorted(lines)

csv_out_file = open(val_csv_out_path, "w")
for class_name in ordered_classes:
    list_videos = os.listdir(os.path.join(dataset_root, split, class_name))
    for video in list_videos:
        if file_type not in video:
            print("Unsupported file type: {}".format(video))
            continue
        write_string = (
            os.path.join(split, class_name, video)
            + separator
            + str(ordered_classes.index(class_name))
            + "\n"
        )
        print(write_string)
        csv_out_file.write(write_string)

csv_out_file.close()
