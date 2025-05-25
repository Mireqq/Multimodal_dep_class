import pandas as pd
import os
import wave
import contextlib
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.io import wavfile
from pydub import AudioSegment


def wav_infos(wav_path):

    with wave.open(wav_path, "rb") as f:
        f = wave.open(wav_path)

        return list(f.getparams())

def read_wav(wav_path):
    with wave.open(wav_path, "rb") as f:

        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]

        # Read the sound data, passing a parameter specifying the length (in sampling points) to be read
        str_data = f.readframes(nframes)

    return str_data

def get_wav_time(wav_path):
    with contextlib.closing(wave.open(wav_path, 'r')) as f:
        frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    return duration


def get_ms_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    start_time = int(start_time)
    end_time = int(end_time)

    sound = AudioSegment.from_mp3(main_wav_path)
    word = sound[start_time:end_time]

    word.export(part_wav_path, format="wav")


def get_second_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    start_time = int(start_time) * 1000
    end_time = int(end_time) * 1000

    sound = AudioSegment.from_mp3(main_wav_path)
    word = sound[start_time:end_time]

    word.export(part_wav_path, format="wav")

def get_minute_part_wav(main_wav_path, start_time, end_time, part_wav_path):
    start_time = (int(start_time.split(':')[0])*60+int(start_time.split(':')[1]))*1000
    end_time = (int(end_time.split(':')[0])*60+int(end_time.split(':')[1]))*1000

    sound = AudioSegment.from_mp3(main_wav_path)
    word = sound[start_time:end_time]

    word.export(part_wav_path, format="wav")


def wav_to_pcm(wav_path, pcm_path)
    f = open(wav_path, "rb")
    f.seek(0)
    f.read(44)

    data = np.fromfile(f, dtype=np.int16)
    data.tofile(pcm_path)

def pcm_to_wav(pcm_path, wav_path):
    f = open(pcm_path,'rb')
    str_data  = f.read()
    wave_out=wave.open(wav_path,'wb')
    wave_out.setnchannels(1)
    wave_out.setsampwidth(2)
    wave_out.setframerate(8000)
    wave_out.writeframes(str_data)


def wav_waveform(wave_path):
    file = wave.open(wave_path)

    a = file.getparams().nframes  # Total number of frames
    f = file.getparams().framerate
    sample_time = 1 / f
    time = a / f
    sample_frequency, audio_sequence = wavfile.read(wave_path)
    # print(audio_sequence)
    x_seq = np.arange(0, time, sample_time)

    plt.plot(x_seq, audio_sequence, 'blue')
    plt.xlabel("time (s)")
    plt.show()

def wav_combine(*args):
    n = args[0][0]  # of wavs to splice
    i = 1
    sounds = []
    while (i <= n):
        sounds.append(AudioSegment.from_wav(args[0][i]))
        i += 1
    playlist = AudioSegment.empty()
    for sound in sounds:
        playlist += sound
    playlist.export(args[0][n + 1], format="wav")


def pdfFilesPath(path):

    filePaths = []  # Name of all files in the storage directory with paths
    for root, dirs, files in os.walk(path):
        for file in files:
            filePaths.append(os.path.join(root, file))
    return filePaths


if __name__ == '__main__':

    path = r'/workspace/audio/'  # Audio directories to be cut
    # The cut audio directory, which is also the audio directory to be merged.
    path_segment = r'/workspace/Depression_Recognition-Code/Preprocessing code/audio_split/'
    path1 = r'/workspace/Depression_Recognition-Code/Preprocessing code/data_time/'
    excel_root = os.listdir(path1)  # excel file directory deposit list



    a_l = []  # Record the number of audio cuts
    print('Start cutting the audio!')

    for root, dir, files in os.walk(path):
        for i in range(len(files)):

            audio = root + files[i]

            time_all = int(get_wav_time(audio) * 1000)

            print("The audio  is %s, the file  is %s" % (files[i], excel_root[i]))

            #
            df = pd.read_excel(path1 + excel_root[i], usecols=['start_time', 'stop_time'])


            index = 1  # Cut serial number names, starting with serial number 1
            k = 0
            # k_l = 0  # Total number of audio cuts recorded
            for j in df:
                l = len(df)
                # print("l= %d" % l)
                while k <= l - 1:

                    start_time = df.loc[k, ['start_time']]
                    end_time = df.loc[k, ['stop_time']]

                    start_time = float(start_time)
                    end_time = float(end_time)

                    aduio_segment = path_segment + files[i][:-4] + '_' + str(index) + '.wav'
                    get_second_part_wav(audio, start_time, end_time, aduio_segment)

                    index += 1
                    k += 1

            a_l.append(k)
            print(a_l)
            print("Cutting out %d audio" % l)

    print('A total of %d audio cuts' % sum(a_l))
    print(a_l)
    print('Audio cutting complete!')

    filePaths = []
    r = []
    for root, dir, files in os.walk(path_segment):
        for file in files:
            f1 = re.findall(r"\d+\.?\d*", file)
            f1 = (f1[0], f1[1][:-1])
            f1 = list(map(int, f1))
            r.append(f1)
        re = sorted(r, key=lambda x: (x[0], x[1]))
        new_files = [(str(re[i][0]) + '_' + 'AUDIO_' + str(re[i][1]) + '.wav') for i in range(len(re))]

        for f2 in new_files:
            filePaths.append(os.path.join(root, f2))

    print('*' * 50)
    print('Start merging audio!')

    # Initial value setting
    n = 5        # Select n audios to merge
    id_num = 300   # Starting number of file naming after merger
    w = 1
    # w1,w2 is the range of audio splices to ensure that only files cut from the same audio are merged,
    # avoiding audio splices such as 300 and 301.
    w1 = 0
    w2 = a_l[0]

    for file in filePaths:
            file_1 = filePaths[w1:w2]
            print("Range of w1, w2：%d %d" % (w1, w2))
            id_num2 = 1

            for b in [file_1[i:i + n] for i in range(0, len(file_1), 5)]:
                out_path = r'/workspace/Depression_Recognition-Code/Preprocessing code/audio_combine/' + str(id_num) + '_' + str(id_num2) + '.wav'
                b.insert(0, len(b))
                b.append(out_path)
                wav_combine(b)
                id_num2 += 1

            if w < len(a_l):
                w1 = w2
                w2 = w2 + a_l[w]
                # 342,394,398,460 not in the dataset file, these numbers are skipped
                if id_num == 341 or id_num == 393 or id_num == 397 or id_num == 459:
                    id_num += 2
                else:
                    id_num += 1
                w += 1
            else:
                break
    print('Audio merge complete!')


