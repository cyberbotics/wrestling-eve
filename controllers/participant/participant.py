# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Demonstrates how to use the camera and gives an image processing example to locate the opponent.
"""

from controller import Robot, Motion
import sys
sys.path.append('..')
from utils.fall_detection import FallDetection # David's fall detection is implemented in this class
from utils.finite_state_machine import FiniteStateMachine
from utils.current_motion_manager import CurrentMotionManager
from utils.running_average import RunningAverage
import utils.image

try:
    import cv2
except ImportError:
    sys.exit("Warning: 'cv2' module not found. Please check the Python modules installation instructions " +
             "at 'https://www.cyberbotics.com/doc/guide/using-python'.")


class Eve (Robot):
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

        # camera
        self.camera = self.getDevice("CameraTop")
        self.camera.enable(self.time_step)

        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)

        # arm motors for getting up from a side fall
        self.RShoulderRoll = self.getDevice("RShoulderRoll")
        self.LShoulderRoll = self.getDevice("LShoulderRoll")

        self.fall_detector = FallDetection(self.time_step, self)

        self.current_motion = CurrentMotionManager()
        # load motion files
        self.motions = {
            'SideStepLeft': Motion('../motions/SideStepLeftLoop.motion'),
            'TurnLeft':     Motion('../motions/TurnLeft20.motion'),
            'TurnRight':    Motion('../motions/TurnRight20.motion'),
        }

        self.opponent_position = RunningAverage(dimensions=1)

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
            [x, y, _] = self.gps.getValues()
            if -0.9 < x < 0.9 and -0.7 < y < 0.7:
                self.current_motion.set(self.motions['SideStepLeft'])
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
        img = utils.image.get_cv_image_from_camera(self.camera)
        largest_contour, vertical, horizontal = self.locate_opponent(img)
        output = img.copy()
        if largest_contour is not None:
            cv2.drawContours(output, [largest_contour], 0, (255, 255, 0), 1)
            output = cv2.circle(output, (horizontal, vertical), radius=2,
                                color=(0, 0, 255), thickness=-1)
        utils.image.send_image_to_robot_window(self, output)
        if horizontal is None:
            return 0
        return horizontal * 2/img.shape[1] - 1

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
        largest_contour = utils.image.get_largest_contour(closing)
        if largest_contour is not None:
            # we get its centroid for an approximate opponent location
            vertical_coordinate, horizontal_coordinate = utils.image.get_contour_centroid(largest_contour)
            return largest_contour, vertical_coordinate, horizontal_coordinate
        else:
            # if no contour is found, we return None
            return None, None, None


# create the Robot instance and run main loop
wrestler = Eve()
wrestler.run()
