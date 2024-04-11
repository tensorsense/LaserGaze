# LaserGaze: Real-Time Gaze Direction Estimation

LaserGaze is an open-source project designed to set a new standard in gaze direction estimation for real-time video streams. Utilizing facial landmark detection, LaserGaze dynamically tracks eye positions and calculates gaze vectors, offering a robust solution for applications in areas such as augmented reality, behavioral analysis, and user interface control. This elevates both performance and accuracy to state-of-the-art (SoTA) levels.

## Key Features

- **Real-Time Processing:** Optimized for live video feeds, not static images.
- **Facial Landmark Utilization:** Uses MediaPipe for cutting-edge facial feature recognition.
- **Dynamic Gaze Estimation:** Continuously updates and refines gaze estimations as new video frames are processed.
- **Customizable Visualization:** Includes options for adjusting the visual representation of gaze vectors on the video output.
- **Asynchronous Support:** Leverages Python's `asyncio` for efficient real-time processing.

## Comparison

Below is a visual comparison that demonstrates the enhancements in gaze direction estimation offered by LaserGaze compared to the previous SoTA solutions. The side-by-side GIFs illustrate the responsiveness and accuracy improvements in real-world scenarios.

<table>
  <tr>
    <td><b>Traditional Frame-by-Frame (SoTA)</b></td>
    <td><b>LaserGaze Implementation</b></td>
  </tr>
  <tr>
    <td><img src="gifs/others.gif" width="320" height="320" alt="Traditional Gaze Tracking"/></td>
    <td><img src="gifs/laser-gaze.gif" width="320" height="320" alt="LaserGaze Gaze Tracking"/></td>
  </tr>
</table>

These visualizations highlight the advancements made by LaserGaze in terms of tracking precision and real-time performance, making it an essential upgrade for any application involving gaze tracking.

## Getting Started

### Prerequisites

- Python 3.7+
- MediaPipe
- OpenCV-Python
- NumPy
- SciPy

You can install all the required packages using:

```bash
pip install -r requirements.txt
```

### Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/tensorsense/LaserGaze.git
cd LaserGaze
```

### Running the Project
To run the project, simply execute the `main.py` script:
```bash
python main.py
```

## Usage Example
The `main.py` file demonstrates a basic setup where `GazeProcessor` is initialized with a callback function that prints out the left and right gaze vectors:
```python
async def gaze_vectors_collected(left, right):
    print(f"left: {left}, right: {right}")

async def main():
    vo = VisualizationOptions()
    gp = GazeProcessor(visualization_options=vo, callback=gaze_vectors_collected)
    await gp.start()
```

## Configuration
Modify the `VisualizationOptions` to change the appearance of the gaze vectors:
```python
vo = VisualizationOptions(color=(255, 0, 0), line_thickness=2)
```

## Contributing
Contributions to LaserGaze are welcome! There are several ways you can contribute:

- **Issue Tracking:** Report bugs or suggest new features.
- **Developing:** Submit your pull requests with new features or bug fixes.
- **Documentation:** Improve or suggest enhancements to the documentation.

## License
This project is licensed under the MIT License

## Acknowledgments
- The LaserGaze team would like to thank the developers of MediaPipe for their state-of-the-art facial landmark detection technology.
- This project is inspired by the need for more interactive and intuitive user interfaces in software applications.

## Technical Overview

LaserGaze integrates several advanced techniques and methodologies to achieve high precision in real-time gaze direction estimation:

- **Affine Transformations with 3D Face Models**: LaserGaze uses affine transformations to align detected facial landmarks from MediaPipe with a pre-defined 3D face model. This allows for precise orientation and positioning of facial features across different frames and camera perspectives, ensuring accurate gaze vector calculations.

- **Dynamic Eye Center Estimation**: The `EyeballDetector` class dynamically estimates the centers of the eyes using sets of 3D points around the eye regions captured through MediaPipe. The system adapts to changes such as eye movements and varying head positions, continuously refining these estimates for enhanced accuracy.

- **Real-Time Video Processing with MediaPipe**: Utilizing MediaPipe's capabilities, LaserGaze processes video streams (not static images) for temporal consistency and smoother tracking. This provides robust, real-time detection of high-fidelity facial landmarks necessary for accurate tracking.

This holistic approach, combining computational geometry, dynamic updating mechanisms, and advanced video processing, establishes LaserGaze as a state-of-the-art tool in real-time gaze estimation, suitable for applications ranging from augmented reality to behavioral analysis.


## Support
For support, please open an issue on the GitHub project page or contact the maintainers directly.

Happy Tracking!