# Humanoid Robot Wrestling Controller Example

[![webots.cloud - Competition](https://img.shields.io/badge/webots.cloud-Competition-007ACC)][1]

## Eve controller

Demonstrates how to use the camera and gives an image processing example to locate the opponent. A step by step explanation of the whole image processing workflow is provided by a [notebook](./notebook/image_processing_explanation.ipynb).

This controller needs OpenCV and Numpy, therefore the [Dockerfile](controllers/Dockerfile#L4-L9) needs to be updated:

```Dockerfile
# Dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    opencv-python
```

Beats [David](https://github.com/cyberbotics/wrestling-david) by locating and dodging him.

Here is the [participant.py](./controllers/participant/participant.py) file:

``` Python
import sys
sys.path.append('..')
from utils.camera import Camera
from utils.fall_detection import FallDetection  # David's fall detection is implemented in this class
from utils.running_average import RunningAverage
from utils.image_processing import ImageProcessing as IP
from utils.finite_state_machine import FiniteStateMachine
from utils.current_motion_manager import CurrentMotionManager
from controller import Robot, Motion
import cv2


class Eve (Robot):
    NUMBER_OF_DODGE_STEPS = 10

    def __init__(self):
        Robot.__init__(self)

        # retrieves the WorldInfo.basicTimeTime (ms) from the world file
        self.time_step = int(self.getBasicTimeStep())

        self.fsm = FiniteStateMachine(
            states=['CHOOSE_ACTION', 'BLOCKING_MOTION'],
            initial_state='CHOOSE_ACTION',
            actions={
                'CHOOSE_ACTION': self.choose_action,
                'BLOCKING_MOTION': self.pending
            }
        )

        self.camera = Camera(self)

        # arm motors for getting up from a side fall
        self.RShoulderRoll = self.getDevice("RShoulderRoll")
        self.LShoulderRoll = self.getDevice("LShoulderRoll")

        self.fall_detector = FallDetection(self.time_step, self)
        self.current_motion = CurrentMotionManager()
        # load motion files
        self.motions = {
            'SideStepLeft': Motion('../motions/SideStepLeftLoop.motion'),
            'SideStepRight': Motion('../motions/SideStepRightLoop.motion'),
            'TurnRight': Motion('../motions/TurnRight20.motion'),
            'TurnLeft': Motion('../motions/TurnLeft20.motion'),
        }
        self.opponent_position = RunningAverage(dimensions=1)
        self.dodging_direction = 'left'
        self.counter = 0

    def run(self):
        while self.step(self.time_step) != -1:
            self.opponent_position.update_average(
                self._get_normalized_opponent_horizontal_position())
            self.fall_detector.check()
            self.fsm.execute_action()

    def choose_action(self):
        if self.opponent_position.average < -0.4:
            self.current_motion.set(self.motions['TurnLeft'])
        elif self.opponent_position.average > 0.4:
            self.current_motion.set(self.motions['TurnRight'])
        else:
            # dodging by alternating between left and right side steps to avoid easily falling off the ring
            if self.dodging_direction == 'left':
                if self.counter < self.NUMBER_OF_DODGE_STEPS:
                    self.current_motion.set(self.motions['SideStepLeft'])
                    self.counter += 1
                else:
                    self.dodging_direction = 'right'
            elif self.dodging_direction == 'right':
                if self.counter > 0:
                    self.current_motion.set(self.motions['SideStepRight'])
                    self.counter -= 1
                else:
                    self.dodging_direction = 'left'
            else:
                return
        self.fsm.transition_to('BLOCKING_MOTION')

    def pending(self):
        # waits for the current motion to finish before doing anything else
        if self.current_motion.is_over():
            self.fsm.transition_to('CHOOSE_ACTION')

    def _get_normalized_opponent_horizontal_position(self):
        """Returns the horizontal position of the opponent in the image, normalized to [-1, 1]
            and sends an annotated image to the robot window."""
        img = self.camera.get_image()
        largest_contour, vertical, horizontal = self.locate_opponent(img)
        output = img.copy()
        if largest_contour is not None:
            cv2.drawContours(output, [largest_contour], 0, (255, 255, 0), 1)
            output = cv2.circle(output, (horizontal, vertical), radius=2,
                                color=(0, 0, 255), thickness=-1)
        self.camera.send_to_robot_window(output)
        if horizontal is None:
            return 0
        return horizontal * 2 / img.shape[1] - 1

    def locate_opponent(self, img):
        """Image processing demonstration to locate the opponent robot in an image."""
        # we suppose the robot to be located at a concentration of multiple color changes (big Laplacian values)
        laplacian = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
        # those spikes are then smoothed out using a Gaussian blur to get blurry blobs
        blur = cv2.GaussianBlur(laplacian, (0, 0), 2)
        # we apply a threshold to get a binary image of potential robot locations
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
        # the binary image is then dilated to merge small groups of blobs together
        closing = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
        # the robot is assumed to be the largest contour
        largest_contour = IP.get_largest_contour(closing)
        if largest_contour is not None:
            # we get its centroid for an approximate opponent location
            vertical_coordinate, horizontal_coordinate = IP.get_contour_centroid(
                largest_contour)
            return largest_contour, vertical_coordinate, horizontal_coordinate
        else:
            # if no contour is found, we return None
            return None, None, None


# create the Robot instance and run main loop
wrestler = Eve()
wrestler.run()
```

[Fatima](https://github.com/cyberbotics/wrestling-fatima) is a more advanced robot controller able to win against Eve.

[1]: https://webots.cloud/run?version=R2022b&url=https%3A%2F%2Fgithub.com%2Fcyberbotics%2Fwrestling%2Fblob%2Fmain%2Fworlds%2Fwrestling.wbt&type=competition "Leaderboard"
