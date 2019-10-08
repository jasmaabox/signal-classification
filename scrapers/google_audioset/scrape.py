import youtube_dl
import os
import scipy.io.wavfile

def main():
    # Collect rows
    target_classes = {
        1:"",            #  1
        2:"/m/02rlv9",   #  2 - Motorboat, speedboat
        3:"/m/0912c9",   #  3 - Vehicle horn, car horn, honking
        4:"/t/dd00130", #  4 - Car ignition
        5:"/m/07qrkrw",  #  5 - Meow
        6:"",            #  6
        7:"/m/01xq0k1", #  7 - Cow
        8:"/m/09xqv",    #  8 - Cricket
        9:"/m/05tny_",   #  9 - Bark
        10:"",           # 10
        11:"",           # 11
        12:"",           # 12
        13:"/m/0g6b5",   # 13 - Fireworks
        14:"/m/07st88b", # 14 - Croak
        15:"/m/03k3r",   # 15 - Horse
        16:"",           # 16
        17:"/m/03p19w",  # 17 - Jackhammer
        18:"/m/04229",   # 18 - Jet engine
        19:"",           # 19
        20:"",           # 20
        21:"/m/07r_80w", # 21 - Hoot
        22:"/m/04qvtq",  # 22 - Police car (siren)
        23:"/m/0_ksk",   # 23 - Power tool
        24:"/m/07bgp",   # 24 - Sheep
        25:"/m/0h9mv",   # 25 - Tire squeal
        26:"/m/0dgbq",   # 26 - Civil defense siren
        27:"/m/01g50p",  # 27 - Railroad car, train wagon
        28:"/m/09x0r",   # 28 - Speech
    }
    target_rows = {}
    for _, c in target_classes.items():
        target_rows[c] = []
    with open('balanced_train_segments.csv', 'r') as f:
        for line in f.readlines():
            v_id, start, end, classes = line.split(", ")
            start = float(start)
            end = float(end)
            classes = classes[1:-2].split(",")

            for _, c in target_classes.items():
                if c in classes:
                    target_rows[c].append( (v_id, start, end) )

    # Download
    for i, c in target_classes.items():

        try:
            os.mkdir(f'data/{i}')
        except FileExistsError:
            print(f'data/{i} already exists')
            
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'data/{i}/%(id)s.%(ext)s',
            'quiet': True,
            'ignoreerrors': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            for v_id, start, end in target_rows[c]:
                ydl.download([f'https://www.youtube.com/watch?v={v_id}'])

                # trim audio
                try:
                    sample_rate, audio_data = scipy.io.wavfile.read(f'data/{i}/{v_id}.wav')
                    cropped_audio_data = audio_data[int(start*sample_rate):int(end*sample_rate)]
                    scipy.io.wavfile.write(f'data/{i}/{v_id}.wav', sample_rate, cropped_audio_data)
                except FileNotFoundError:
                    print(f'data/{i}/{v_id}.wav not found')

if __name__ == "__main__":
    main()
