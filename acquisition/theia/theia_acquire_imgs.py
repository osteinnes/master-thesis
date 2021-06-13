import os
import asyncio
import cvb
import time
import numpy as np

rate_counter = None
abort = False

async def async_acquire():
    global rate_counter
    with cvb.DeviceFactory.open(os.path.join(cvb.install_path(), "drivers", "GenICam.vin"), port=0) as device:
        with cvb.DeviceFactory.open(os.path.join(cvb.install_path(), "drivers", "GenICam.vin"), port=1) as device0:
            
            # Init and open stream
            stream = device.stream
            stream0 = device0.stream
            stream.start()
            stream0.start()

            # Create counter
            rate_counter = cvb.RateCounter()

            im = 1
            while not abort:
                
                # Wait for external trigger
                result = await  stream.wait_async()
                time1 = time.time()
                result0 = await stream0.wait_async()
                time2 = time.time()

                # Fetch result (image and status)
                image, status = result.value
                image0, status0 = result0.value

                # Increment counter
                rate_counter.step()
                
                # Check if both acquired frames are ok
                # Save if ok
                if status == cvb.WaitStatus.Ok and status0 == cvb.WaitStatus.Ok:
                    leftim_string = "left_%s.bmp" % im
                    rightim_string = "right_%s.bmp" % im

                    image.save(leftim_string)
                    image0.save(rightim_string)
                    
                    im+=1                  

            stream.abort()
            stream0.abort()
    

if __name__ == '__main__':

    watch = cvb.StopWatch()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(
        async_acquire())) 
    loop.close()

    duration = watch.time_span

    print("Acquired on port 0 with " + str(rate_counter.rate) + " fps")
    print("Overall measurement time: " +str(duration / 1000) + " seconds")


