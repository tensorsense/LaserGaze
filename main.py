# -----------------------------------------------------------------------------------
# Company: TensorSense
# Project: LaserGaze
# File: main.py
# Description: This script demonstrates a basic example of how to use the GazeProcessor class
#              from the LaserGaze project. It sets up the gaze detection system with
#              optional visualization settings and an asynchronous callback for processing
#              gaze vectors. The example provided here can be modified or extended by
#              contributors to fit specific needs or to experiment with different settings
#              and functionalities. It serves as a starting point for developers looking to
#              integrate and build upon the gaze tracking capabilities provided by the
#              GazeProcessor in their own applications.
# Author: Sergey Kuldin
# -----------------------------------------------------------------------------------

from GazeProcessor import GazeProcessor
from VisualizationOptions import VisualizationOptions
import asyncio

async def gaze_vectors_collected(left, right):
    print(f"left: {left}, right: {right}")

async def main():
    vo = VisualizationOptions()
    gp = GazeProcessor(visualization_options=vo, callback=gaze_vectors_collected)
    await gp.start()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()