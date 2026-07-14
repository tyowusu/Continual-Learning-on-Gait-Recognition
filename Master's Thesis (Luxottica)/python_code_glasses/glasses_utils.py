import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    timestamps = []
    slow_data = []
    fast_data = []

    c = 1
    with open(path) as csvfile:
        file_reader = csv.reader(csvfile, delimiter = '\t')
        for row in file_reader:
            if c == 1:
                timestamps.append(row[0])
            if c == 24: 
                slow_header = [i.strip() for i in row]
            if c > 24 and c < 28:
                slow_data.append([float(i) for i in row])
            if c == 28:
                fast_header = [i.strip() for i in row]
            if c > 28 and c < 44:
                fast_data.append([float(i) for i in row])
            if c == 43:
                c = 0            
            c = c+1

            

    if len(row) == 1:
        timestamps.pop(-1)

    interval = row[0]
    
    slow_df = pd.DataFrame(slow_data, columns = slow_header)
    fast_df = pd.DataFrame(fast_data, columns = fast_header)

    slow_df['timestamps']= np.repeat(timestamps,3)
    fast_df['timestamps']= np.repeat(timestamps,15)
    
    return fast_df, slow_df, interval



def plot_data(fast_df, slow_df, filepath):
    # PLOTS
    fig, ((ax1, ax3, ax5, ax7), (ax2, ax4, ax6, ax8)) = plt.subplots(ncols=4, nrows=2, constrained_layout=True)

    ax1.plot(fast_df['accx'], 'r', label = 'accx')
    ax1.plot(fast_df['accy'], 'g', label = 'accy')
    ax1.plot(fast_df['accz'], 'b', label = 'accz')
    ax1.set_title('Acceleration', fontsize=10)
    ax1.legend()
    #ax1.set_xlabel('Time (s)')
    #ax1.set_ylabel('PPG(%)')

    ax2.plot(fast_df['gyrx'], 'r', label = 'gyrx')
    ax2.plot(fast_df['gyry'], 'g', label = 'gyry')
    ax2.plot(fast_df['gyrz'], 'b', label = 'gyrz')
    ax2.set_title('Angular velocity', fontsize=10)
    ax2.legend()
    #ax2.set_xlabel('Time (s)')
    #ax2.set_ylabel('SpO2(%)')
    #ax2.set_ylim([np.min([np.min(spo2_signal)-1, 95]), 101])

    ax3.plot(fast_df['pitch'], 'r', label = 'pitch')
    ax3.plot(fast_df['yaw'], 'g', label = 'yaw')
    ax3.plot(fast_df['roll'], 'b', label = 'roll')
    ax3.set_title('Angular position', fontsize=10)
    ax3.legend()
    #ax3.set_xlabel('Time (s)')
    #ax3.set_ylabel('ECG(mV)')

    ax4.plot(fast_df['magx'], 'r', label = 'magx')
    ax4.plot(fast_df['magy'], 'g', label = 'magy')
    ax4.plot(fast_df['magz'], 'b', label = 'magz')
    ax4.set_title('Magnetometer', fontsize=10)
    ax4.legend()
    #ax1.set_xlabel('Time (s)')
    #ax1.set_ylabel('PPG(%)')

    ax5.plot(slow_df['rh'], 'r', label = 'Relative humidity')
    ax5.plot(slow_df['temp'], 'g', label = 'temperature')
    ax5.set_title('Environmental info', fontsize=10)
    ax5.legend()
    #ax2.set_xlabel('Time (s)')
    #ax2.set_ylabel('SpO2(%)')
    #ax2.set_ylim([np.min([np.min(spo2_signal)-1, 95]), 101])

    ax6.plot(slow_df['pressure'], 'b', label = 'pressure')
    ax6.set_title('Pressure', fontsize=10)
    #ax2.set_xlabel('Time (s)')
    #ax2.set_ylabel('SpO2(%)')
    #ax2.set_ylim([np.min([np.min(spo2_signal)-1, 95]), 101])

    ax7.plot(slow_df['uva'], 'k', label = 'UV A')
    ax7.plot(slow_df['uvb'], 'c', label = 'UV B')
    ax7.set_title('UV light', fontsize=10)
    ax7.legend()
    #ax3.set_xlabel('Time (s)')
    #ax3.set_ylabel('ECG(mV)')

    ax8.plot(slow_df['x'], 'cyan', label = 'X')
    ax8.plot(slow_df['y'], 'deepskyblue', label = 'Y')
    ax8.plot(slow_df['blueb'], 'darkblue', label = 'Blue bad')
    ax8.plot(slow_df['blueg'], 'royalblue', label = 'Blue good')
    ax8.set_title('Blue light', fontsize=10)
    ax8.legend()
    #ax3.set_xlabel('Time (s)')
    #ax3.set_ylabel('ECG(mV)')

    '''
    # Adherence works bad!!!!!
    ax10.plot(slow_df['worn'], 'k', label = 'worn')
    ax10.set_title('Adherence', fontsize=10)
    #ax3.set_xlabel('Time (s)')
    #ax3.set_ylabel('ECG(mV)')
    '''

    fig.suptitle(filepath, fontsize=13)

    # Zoom and play with your plot!
    plt.show()
