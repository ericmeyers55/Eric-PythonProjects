import os
import struct
from datetime import datetime
from datetime import timedelta
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import time
import pandas as pd

class LoadReplayData:
    def __init__(self, filename):
        self.filename = filename

        self.read_meta_data()

    def read_meta_data(self):
        # open the file and parse through to grab the meta data
        with open(self.filename, 'rb') as f:
            # seek to beginning of file and read the first 4 bytes: version number
            f.seek(0)
            self.version = LoadReplayData.unpack_byte_array(f.read(4), 'int32')

            # Read in the subject id
            temp_length = LoadReplayData.unpack_byte_array(f.read(4), 'int32')
            self.subject_id = f.read(temp_length).decode()

            # read in the game id
            temp_length = LoadReplayData.unpack_byte_array(f.read(4), 'int32')
            self.game_id = f.read(temp_length).decode()

            # read in exercise id
            temp_length = LoadReplayData.unpack_byte_array(f.read(4), 'int32')
            self.exercise_id = f.read(temp_length).decode()

            # read in device type
            temp_length = LoadReplayData.unpack_byte_array(f.read(4), 'int32')
            self.device_type = f.read(temp_length).decode()

            # read in data type. 0: Controller data; 1: Game Data
            self.data_type = LoadReplayData.unpack_byte_array(f.read(1), 'uint8')

            # read in the start time
            temp_time = LoadReplayData.unpack_byte_array(f.read(8), 'float64')
            self.session_start = LoadReplayData.convert_datenum_to_dateTime(temp_time)

            # data location starts here
            self.data_start_location = f.tell()

            # seek to the end of file and read the final 8 bytes
            f.seek(-8, 2)

            # grab the file position for end of data
            self.data_end_location = f.tell()

            # grab the number of data samples in the file
            self.total_frames = LoadReplayData.unpack_byte_array(f.read(4), 'int32')

            # grab the number of stimulations in the file
            self.total_stimulations = LoadReplayData.unpack_byte_array(f.read(4), 'int32')

    def read_data(self):
        if self.data_type == 0:
            self.read_controller_data()

        elif self.data_type == 1:
            if self.game_id == 'FruitArchery':
                self.read_fruit_archery_data()
            else:
                pass
        else:
            pass

    def read_fruit_archery_data(self):
        # pre-allocate all of the numpy arrays
        self.sample_timenums = np.zeros((self.total_frames, 1))
        self.sample_time = np.zeros((self.total_frames, 1))

        self.arrow_exists = np.zeros((self.total_frames, 1))
        self.arrow_flying = np.zeros((self.total_frames, 1))
        self.arrow_position = np.zeros((self.total_frames, 2))
        self.arrow_velocity = np.zeros((self.total_frames, 2))

        self.bow_exists = np.zeros((self.total_frames, 1))
        self.bow_rotation = np.zeros((self.total_frames, 1))
        self.bow_position = np.zeros((self.total_frames, 2))

        self.fruit_exists = np.zeros((self.total_frames, 1))
        self.fruit_position = np.zeros((self.total_frames, 2))
        self.fruit_rotation = np.zeros((self.total_frames, 1))
        self.fruit_size = np.zeros((self.total_frames, 2))
        self.fruit_hit_byarrow = np.zeros((self.total_frames, 1))
        self.score = np.zeros((self.total_frames, 1))

        with open(self.filename, 'rb') as f:

            # seek to the position in the file to begin reading trial information
            f.seek(self.data_start_location)

            self.stage_number = LoadReplayData.unpack_byte_array(f.read(4), 'int')

            for sample in range(self.total_frames):
                self.sample_timenums[sample] = LoadReplayData.unpack_byte_array(f.read(8), 'float64')
                self.sample_time[sample] = LoadReplayData.calculate_timedelta_from_datenums(
                    self.sample_timenums.item(0), self.sample_timenums.item(sample))

                # Read in the arrow information
                self.arrow_exists[sample] = LoadReplayData.unpack_byte_array(f.read(1), 'uint8')
                if self.arrow_exists[sample] == 1:
                    self.arrow_flying[sample] = LoadReplayData.unpack_byte_array(f.read(1), 'uint8')
                    if self.arrow_flying[sample] == 1:
                        self.arrow_position[sample,0] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                        self.arrow_position[sample,1] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                        self.arrow_velocity[sample,0] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                        self.arrow_velocity[sample,1] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                    else:
                        self.arrow_position[sample, :] = np.nan
                        self.arrow_velocity[sample, :] = np.nan
                else:
                    self.arrow_flying[sample] = np.nan
                    self.arrow_position[sample,:] = np.nan
                    self.arrow_velocity[sample,:] = np.nan

                # read in the bow information
                self.bow_exists[sample] = LoadReplayData.unpack_byte_array(f.read(1), 'uint8')
                if self.bow_exists[sample] == 1:
                    self.bow_position[sample, 0] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                    self.bow_position[sample, 1] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                    self.bow_rotation[sample] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                else:
                    self.bow_position[sample,:] = np.nan
                    self.bow_rotation[sample] = np.nan

                # Read in the fruit information
                self.fruit_exists[sample] = LoadReplayData.unpack_byte_array(f.read(1), 'uint8')
                if self.fruit_exists[sample] == 1:
                    self.fruit_position[sample, 0] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                    self.fruit_position[sample, 1] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                    self.fruit_rotation[sample] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                    self.fruit_size[sample, 0] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                    self.fruit_size[sample, 1] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                    self.fruit_hit_byarrow[sample] = LoadReplayData.unpack_byte_array(f.read(1), 'uint8')
                else:
                    self.fruit_position[sample,:] = np.nan
                    self.fruit_rotation[sample] = np.nan
                    self.fruit_size[sample,:] = np.nan
                    self.fruit_hit_byarrow[sample] = np.nan

                # Read in the score
                self.score[sample] = LoadReplayData.unpack_byte_array(f.read(4), 'int32')

    def read_controller_data(self):

        # pre-allocate all of the numpy arrays
        self.sample_timenums = np.zeros((self.total_frames, 1))
        self.sample_time = np.zeros((self.total_frames, 1))

        if self.device_type == 'FitMi':
            self.gyro = np.zeros((self.total_frames, 3, 2))
            self.acc = np.zeros((self.total_frames, 3, 2))
            self.mag = np.zeros((self.total_frames, 3, 2))
            self.quat = np.zeros((self.total_frames, 4, 2))
            self.loadcell = np.zeros((self.total_frames, 1, 2))
            self.touch = np.zeros((self.total_frames, 1, 2))
            self.battery = np.zeros((self.total_frames, 1, 2))

        elif self.device_type == 'Touchscreen':
            self.touch_position = np.zeros((self.total_frames, 2))

        # create temporary list to store the samples that VNS stims occur
        stim_samples = []
        stim_times = []

        # Open the file and loop through
        with open(self.filename, 'rb') as f:

            # seek to the position in the file to begin reading trial information
            f.seek(self.data_start_location)

            # read the packet identifier
            id_packet = LoadReplayData.unpack_byte_array(f.read(4), 'int32')

            # data from one frame should come in the following order:
            #Puck data then (if stim occurs) stimulation time
            for sample in range(self.total_frames):

                # if this is a puck data packet id = 1
                if id_packet == 1:
                    self.sample_timenums[sample] = LoadReplayData.unpack_byte_array(f.read(8), 'float64')
                    self.sample_time[sample] = LoadReplayData.calculate_timedelta_from_datenums(
                        self.sample_timenums.item(0), self.sample_timenums.item(sample))

                    # Loop through both pucks
                    for pucknum in range(0, 2):

                        #skip over the next puck identifier int32
                        f.seek(4, 1)

                        # Read the 3 accelerometer values
                        for i in range(0, 3):
                            self.acc[sample, i, pucknum] = LoadReplayData.unpack_byte_array(f.read(4), 'int32')

                        # read the 3 gyro values
                        for i in range(0,3):
                            self.gyro[sample, i, pucknum] = LoadReplayData.unpack_byte_array(f.read(4), 'int32')

                        # read the 3 magnetometer values
                        for i in range(0, 3):
                            self.mag[sample, i, pucknum] = LoadReplayData.unpack_byte_array(f.read(8), 'float64')

                        # read the 4 quaternion values
                        for i in range(0, 4):
                            self.quat[sample, i, pucknum] = LoadReplayData.unpack_byte_array(f.read(8), 'float64')

                        self.loadcell[sample, 0, pucknum] = LoadReplayData.unpack_byte_array(f.read(4), 'int32')
                        self.touch[sample, 0, pucknum] = LoadReplayData.unpack_byte_array(f.read(1), 'int8')
                        self.battery[sample, 0, pucknum] = LoadReplayData.unpack_byte_array(f.read(4), 'int32')

                #Else if this is a touchscreen packet id = 4
                elif id_packet == 4:
                    self.sample_timenums[sample] = LoadReplayData.unpack_byte_array(f.read(8), 'float64')
                    self.sample_time[sample] = LoadReplayData.calculate_timedelta_from_datenums(
                        self.sample_timenums.item(0), self.sample_timenums.item(sample))
                    self.touch_position[sample, 0] = LoadReplayData.unpack_byte_array(f.read(4), 'float')
                    self.touch_position[sample, 1] = LoadReplayData.unpack_byte_array(f.read(4), 'float')

                # read the packet identifier
                id_packet = LoadReplayData.unpack_byte_array(f.read(4), 'int32')

                # if  this is a stim packet, don't increment the for counter and add in to same sample as previous
                if id_packet == 3:
                    stim_samples.append(sample)
                    stim_times.append(LoadReplayData.unpack_byte_array(f.read(8), 'float64'))


            # Convert the stim_samples to a numpy array
            self.stims_samples = np.array(stim_samples)
            self.stim_times = np.array(stim_times)

    # this method converts to the unicode literal for type unpacking from bytes
    # https://docs.python.org/3/library/struct.html
    @staticmethod
    def unpack_byte_array(byte_to_convert, desired_type):
        typedict = {'char': 'c',
                    'int': 'i',
                    'int32': 'i',
                    'int8': 'b',
                    'unsigned int': 'I',
                    'uint8': 'B',
                    'float': 'f',
                    'float64': 'd',
                    'double': 'd'}
        unpacked = struct.unpack(typedict[desired_type], byte_to_convert)
        return unpacked[0]


    #static methods don't take the instance or the class as the first argument
    #if you don't access the instance or the class anywhere within the function, it should be static
    @staticmethod
    def convert_datenum_to_dateTime(datenum):
        """
        Convert Matlab datenum into Python datetime.
        :param datenum: Date in datenum format
        :return:        Datetime object corresponding to datenum.
        """
        days = datenum % 1
        return datetime.fromordinal(int(datenum)) \
               + timedelta(days=days) \
               - timedelta(days=366)

    # This method finds the time difference of two datenums (final_datenum - initial_datenum)
    @staticmethod
    def calculate_timedelta_from_datenums(initial_datenum, final_datenum):
        initial_datetime = LoadReplayData.convert_datenum_to_dateTime(initial_datenum)
        final_datetime = LoadReplayData.convert_datenum_to_dateTime(final_datenum)
        delta = (final_datetime - initial_datetime).total_seconds()
        return delta





controller = LoadReplayData('Z:\Eric\RePlay test data\TestSubject_FruitArchery_20190620_152717.txt')
game = LoadReplayData('Z:\Eric\RePlay test data\TestSubject_FruitArchery_20190620_152717_gamedata.txt')

controller.read_data()
game.read_data()

LoadReplayData.calculate_timedelta_from_datenums



fig, ax=plt.subplots()
# ax.plot(controller.touch_position[:,0], controller.touch_position[:,1])
ax.plot(range(len(controller.loadcell[:,0,1])), controller.loadcell[:,0,0])

ax.set(xlabel='samples', ylabel='Loadcell Value', title='Fruit Archery Gameplay')

# directory = r'Z:\Eric\RePlay test data'

# list_of_files = []
# for dirpath, dirnames, filenames in os.walk(directory):
#     for file in filenames:
#         if file.endswith('GameData.txt'):
#             list_of_files.append(os.path.join(dirpath, file))


# replayfile = LoadReplayData('Z:\Eric\RePlay test data\TestSubject_FruitArchery_20190603_171309_gamedata.txt')
# replayfile.read_fruit_archery_data()

# puckfile = LoadReplayData('Z:\Eric\RePlay test data\TestSubject_FruitArchery_20190603_171309.txt')
# puckfile.read_controller_data()


# print("---%s seconds ---" % (time.time() - start_time))


# temp = replayfile.sample_timenums.flatten()
# timestamps = pd.to_datetime(temp-719529, unit='D', box='False')

# start_time = time.time()