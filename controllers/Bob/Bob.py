# Copyright 1996-2022 Cyberbotics Ltd.
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

"""Controller example for the Robot Wrestling Tournament.
   Demonstrates how to access sensors and actuators"""

from controller import Robot, Motion


class Wrestler (Robot):
    def __init__(self):
        Robot.__init__(self)
        self.timeStep = int(self.getBasicTimeStep())  # retrieves the WorldInfo.basicTimeTime (ms) from the world file

        # camera
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        # there are 7 controllable LEDs on the NAO robot, but we will use only the ones in the eyes
        self.leds = []
        self.leds.append(self.getDevice('Face/Led/Right'))
        self.leds.append(self.getDevice('Face/Led/Left'))

        # shoulder motors
        self.RShoulderPitch = self.getDevice("RShoulderPitch")
        self.RShoulderRoll = self.getDevice("RShoulderRoll")
        self.LShoulderPitch = self.getDevice("LShoulderPitch")
        self.LShoulderRoll = self.getDevice("LShoulderRoll")

        # load motion files
        self.forwards = Motion('../motions/Forwards50.motion')
        self.turnLeft60 = Motion('../motions/TurnLeft60.motion')

    def run(self):
        self.RShoulderPitch.setPosition(1.57)  # arms down
        self.RShoulderRoll.setPosition(-0.15)
        self.LShoulderPitch.setPosition(1.57)
        self.LShoulderRoll.setPosition(0.15)

        self.forwards.setLoop(True)

        self.leds[0].set(0xff0000)  # set eyes to red
        self.leds[1].set(0xff0000)

        while self.step(self.timeStep) != -1:
            t = self.getTime()
            if t == 1:
                self.forwards.play()
            elif t == 22:
                self.forwards.stop()
                self.turnLeft60.play()
                self.RShoulderPitch.setPosition(0)  # uppercut
            elif t == 23.5:
                self.turnLeft60.stop()
            elif t == 26:
                self.forwards.play()  # push
            elif t == 29:
                self.forwards.stop()  # stop
            elif t == 30:
                self.LShoulderPitch.setPosition(-1.57)  # victory
                self.RShoulderPitch.setPosition(-1.57)
                self.leds[0].set(0x00ff00)  # set eyes to green
                self.leds[1].set(0x00ff00)
            pass


# create the Robot instance and run main loop
wrestler = Wrestler()
wrestler.run()
