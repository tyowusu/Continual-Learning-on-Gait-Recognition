"""
isee_decode.py

    Query an ISEE device and decode raw data continuously for 5 seconds.
     v1 08/02/2022 dino.michelon@luxottica.com

    
    Tested on python 3.7.4
    (C) LUXOTTICA GROUP SPA
"""

import sys
import asyncio
import platform
import time
from struct import *

# force print flush
import functools
print = functools.partial(print, flush=True)

from bleak import BleakClient
from bleak import BleakScanner

# ISEE BLE CHARAC
CHARACTERISTIC_UUID_CTRL = "656eeee2-5cc3-4855-929c-6cda2244296e"
CHARACTERISTIC_UUID_DATA = "656eeee1-5cc3-4855-929c-6cda2244296e"
CHARACTERISTIC_UUID_BATTERY_SERVICE = "00002a19-0000-1000-8000-00805f9b34fb"

DATA_INIT0 = b"\x00\xC8\x01\x03"
DATA_INIT1 = b"\x02\xC8\x01\x03"
DATA_ACK = b"\x03\x96"

raw_data = ""

def decode_data():
    global raw_data
    print(f"Decoding raw data... {round(time.time() * 1000)} ms")
    raw_data = bytes.fromhex(raw_data)

    data_id = raw_data[:2]
    #print(unpack('h', data_id))

    data_format_env = "<hHhhhhhhhhI"
    data_size_env = calcsize(data_format_env)
    data_format_imu = "<12h"
    data_size_imu = calcsize(data_format_imu)

    raw_data = raw_data[4:]
    data_env = raw_data[0:3*data_size_env]
    data_imu = raw_data[3*data_size_env:]


    for rh, temp, uva, uvb, x, y, blueg, blueb, worn, zero, pressure  in iter_unpack(data_format_env, data_env):
        print("RH/T:", rh/100.0, temp/100.0, "UV:", uva*6.35/100.0, uvb*8.52/1000.0, "Blue:", x, y, blueg, blueb, "W:", worn, "P:", pressure)
        
    for accx, accy, accz, gyrx, gyry, gyrz, magx, magy, magz, roll, pitch, yaw in iter_unpack(data_format_imu, data_imu):
        print("ACC:", accx, accy, accz, "GYR:", gyrx/10, gyry/10, gyrz/10, "MAG:", magx/10, magy/10, magz/10, "Attitude:", roll/10, pitch/10, yaw/10)  
    
    raw_data = ""
  

async def main(mac_addr: str):
    
    async with BleakClient(mac_addr) as client:
    
        async def notification_handler_data(sender, data):
            global raw_data
            #print("DATA {0}: {1}".format(sender, data.hex()))

            raw_data = raw_data + data.hex()[2:]
            if(data.hex()[0:2] == "96"):
                decode_data()
                # Send ACK
                await client.write_gatt_char(CHARACTERISTIC_UUID_CTRL, DATA_ACK, True)
                
            
        async def notification_handler_control(sender, data):
            print("CTL {0}:: {1}".format(sender, data.hex()))

        
        def notification_handler_battery(sender, data):
            print("BAT {0}:: {1}".format(sender, data.hex()))
            
            
        print(f"Connected: {client.is_connected}")
   

        battery = await client.read_gatt_char(CHARACTERISTIC_UUID_BATTERY_SERVICE)
        print(f"Battery: {int(battery[0]):d}")
        
        await client.start_notify(CHARACTERISTIC_UUID_DATA, notification_handler_data)
        await client.start_notify(CHARACTERISTIC_UUID_CTRL, notification_handler_control)
        #await client.start_notify(CHARACTERISTIC_UUID_BATTERY_SERVICE, notification_handler_battery)

        # Init data trans
        await client.write_gatt_char(CHARACTERISTIC_UUID_CTRL, DATA_INIT0)
        await client.write_gatt_char(CHARACTERISTIC_UUID_CTRL, DATA_INIT1)

        # Wait while data flows
        await asyncio.sleep(5.0)
        
        # End
        await client.stop_notify(CHARACTERISTIC_UUID_CTRL)
        await client.stop_notify(CHARACTERISTIC_UUID_DATA)
        #await client.stop_notify(CHARACTERISTIC_UUID_BATTERY_SERVICE)
          

if __name__ == "__main__":

    if len(sys.argv) == 2:
        ADDRESS = sys.argv[1]
    else:
        ADDRESS =  "C5:F9:A6:3B:41:02"
    asyncio.run(main(ADDRESS))