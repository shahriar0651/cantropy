#!/bin/bash

# MIT License
#
# Copyright (c) 2023 Md Hasan Shahriar
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Project: CANtropy - Time Series Feature Extraction-Based Intrusion Detection Systems for Controller Area Networks
# Author: Md Hasan Shahriar
# Email: hshahriar@vt.edu
#

DIRECTORY="../datasets/can-ids/syncan/"
# Downloading syncan dataset from the github repo
if [ -d "$DIRECTORY" ]; then
  echo "The folder $DIRECTORY already exists." 
  echo "To update the dataset please delete $DIRECTORY and run this script again."
else
  git clone https://github.com/etas/SynCAN.git "$DIRECTORY"
  echo "Raw SynCAN dataset downloaded in $DIRECTORY"
  cd ../datasets/can-ids/syncan/
  unzip 'train_*.zip' -d ambients
  echo "Unzipped training dataset in datasets/can-ids/syncan/ambients"
  unzip 'test_*.zip' -d attacks
  echo "Unzipped training dataset in datasets/can-ids/syncan/attacks" 
  rm -rf *.zip
  rm -rf attacks/test_normal*
  echo "SyncCAN Data Downloaded!"
fi