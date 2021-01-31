import numpy as np
import os
import pickle as pickle
import xml.etree.ElementTree as ET


# create data file from raw xml files from iam handwriting source.
# noinspection PyMethodMayBeStatic
class DataParser:
    def __init__(self, logger):
        self.logger = logger
        self.text_temp = {}

    def run(self, stroke_dir, ascii_dir, data_file):
        self.logger.write("\tparsing dataset...")
        strokes = []
        asciis = []
        for stroke_file in self.__list_xml_files(stroke_dir):
            ascii_file = stroke_file.replace(stroke_dir, ascii_dir)[:-7] + '.txt'
            text_ascii = self.__load_ascii_file(ascii_file, int(stroke_file[-6:-4]) - 1)
            if len(text_ascii) > 10:
                strokes.append(self.__stroke_to_array(self.__load_stroke_file(stroke_file)))
                asciis.append(text_ascii)
            else:
                self.logger.write("\tline length was too short. line was: " + text_ascii)

        assert len(strokes) == len(asciis), "BUG BUG"
        with open(data_file, 'wb') as f:
            pickle.dump([strokes, asciis], f, protocol=2)
        self.logger.write("\tfinished parsing dataset. saved {} lines".format(len(strokes)))

    def __list_xml_files(self, root_dir):
        return [os.path.join(dir_name, name)
                for dir_name, subdir_list, current_list in os.walk(root_dir)
                for name in current_list if name[-3:] == 'xml']

    # function to read each individual xml file
    def __load_stroke_file(self, filename):
        root = ET.parse(filename).getroot()
        x = y = 1e20
        height = 0
        for i in range(1, 4):
            x = min(x, float(root[0][i].attrib['x']))
            y = min(y, float(root[0][i].attrib['y']))
            height = max(height, float(root[0][i].attrib['y']))
        height -= y
        x -= 100
        y -= 100
        return [[[float(p.attrib['x']) - x, float(p.attrib['y']) - y] for p in stroke.findall('Point')]
                for stroke in root[1].findall('Stroke')]

    def __load_ascii_file(self, file, line):
        if file not in self.text_temp:
            with open(file, 'r') as f:
                s = f.read()
            self.text_temp[file] = lines = s[s.find('CSR'):].split('\n')
        else:
            lines = self.text_temp[file]
        return lines[line + 2] if len(lines) > line + 2 else ''

    # converts a list of arrays into a 2d numpy int16 array
    def __stroke_to_array(self, stroke):
        n_point = 0
        for i in range(len(stroke)):
            n_point += len(stroke[i])
        stroke_data = np.zeros((n_point, 3), dtype=np.int16)
        prev_x = prev_y = counter = 0
        for j in range(len(stroke)):
            for k in range(len(stroke[j])):
                stroke_data[counter, 0] = int(stroke[j][k][0]) - prev_x
                stroke_data[counter, 1] = int(stroke[j][k][1]) - prev_y
                prev_x = int(stroke[j][k][0])
                prev_y = int(stroke[j][k][1])
                stroke_data[counter, 2] = 1 if k == (len(stroke[j]) - 1) else 0  # 1: end of stroke
                counter += 1
        return stroke_data


class DataLoader:
    def __init__(self, args, logger, limit=500):
        self.alphabet = args.alphabet
        self.batch_size = args.batch_size
        self.tsteps = args.tsteps
        self.data_scale = args.data_scale  # scale data down by this factor
        self.ascii_steps = int(args.tsteps / args.tsteps_per_ascii)
        self.logger = logger
        self.limit = limit  # removes large noisy gaps in the data

        data_dir = args.data_dir
        data_file = os.path.join(data_dir, "strokes_training_data.cpkl")

        if not (os.path.exists(data_file)):
            self.logger.write("\tcreating training data cpkl file from raw source")
            DataParser(logger).run(os.path.join(data_dir, 'lineStrokes'), os.path.join(data_dir, 'ascii'), data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()

    def load_preprocessed(self, data_file):
        self.logger.write("\tloading dataset...")
        with open(data_file, 'rb') as f:
            [raw_stroke_data, raw_ascii_data] = pickle.load(f)
        # goes thru the list, and only keeps the text entries that have more than tsteps points
        # every 1 in 20 (5%) will be used for validation data
        self.logger.write("\tassembling dataset...")
        self.stroke_data = []
        self.ascii_data = []
        self.valid_stroke_data = []
        self.valid_ascii_data = []
        cur_data_counter = 0
        for i, data in enumerate(raw_stroke_data):
            if len(data) > (self.tsteps + 2):
                # removes large gaps from the data and convert to float32
                data = np.array(np.clip(data, -self.limit, self.limit), dtype=np.float32)
                data[:, 0:2] /= self.data_scale
                cur_data_counter += 1
                if cur_data_counter % 20 == 0:
                    self.valid_stroke_data.append(data)
                    self.valid_ascii_data.append(raw_ascii_data[i])
                else:
                    self.stroke_data.append(data)
                    self.ascii_data.append(raw_ascii_data[i])
        # minus 1, since we want the ydata to be a shifted version of x data
        self.num_batches = int(len(self.stroke_data) / self.batch_size)
        self.logger.write("\t\t{} train individual data points".format(len(self.stroke_data)))
        self.logger.write("\t\t{} valid individual data points".format(len(self.valid_stroke_data)))
        self.logger.write("\t\t{} batches".format(self.num_batches))

    # returns validation data
    def validation_data(self):
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in range(self.batch_size):
            valid_ix = i % len(self.valid_stroke_data)
            data = self.valid_stroke_data[valid_ix]
            x_batch.append(np.copy(data[:self.tsteps]))
            y_batch.append(np.copy(data[1:self.tsteps + 1]))
            ascii_list.append(self.valid_ascii_data[valid_ix])
        one_hots = [to_one_hot(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    # returns a randomized, tsteps-sized portion of the training data
    def next_batch(self):
        x_batch = []
        y_batch = []
        ascii_list = []
        for i in range(self.batch_size):
            data = self.stroke_data[self.idx_perm[self.pointer]]
            x_batch.append(np.copy(data[:self.tsteps]))
            y_batch.append(np.copy(data[1:self.tsteps + 1]))
            ascii_list.append(self.ascii_data[self.idx_perm[self.pointer]])
            self.tick_batch_pointer()
        one_hots = [to_one_hot(s, self.ascii_steps, self.alphabet) for s in ascii_list]
        return x_batch, y_batch, ascii_list, one_hots

    def tick_batch_pointer(self):
        self.pointer += 1
        if self.pointer >= len(self.stroke_data):
            self.reset_batch_pointer()

    def reset_batch_pointer(self):
        self.idx_perm = np.random.permutation(len(self.stroke_data))
        self.pointer = 0


# utility function for converting input ascii characters into vectors the network can understand.
# index position 0 means "unknown"
def to_one_hot(s, ascii_steps, alphabet):
    s = s[:3e3] if len(s) > 3e3 else s  # clip super-long strings
    seq = [alphabet.find(char) + 1 for char in s]
    if len(seq) >= ascii_steps:
        seq = seq[:ascii_steps]
    else:
        seq = seq + [0] * (ascii_steps - len(seq))
    one_hot = np.zeros((ascii_steps, len(alphabet) + 1))
    one_hot[np.arange(ascii_steps), seq] = 1
    return one_hot


# abstraction for logging
class Logger:
    def __init__(self, args):
        self.path = os.path.join(args.log_dir, 'train_scribe.txt' if args.train else 'sample_scribe.txt')
        with open(self.path, 'w') as f:
            f.write("Scribe: Realistic Handriting in Tensorflow\n     by Sam Greydanus\n\n\n")

    def write(self, s, print_it=True):
        if print_it:
            print(s)
        with open(self.path, 'a') as f:
            f.write(s + '\n')
