#!/usr/bin/env python3

########################################################################################################################
# To modify SMA labels used for SMA labeller model
# Author - Timothy Hendrickson

########################################################################################################################

import boto3
from nipype.interfaces.fsl.utils import ConvertXFM
from nipype.interfaces.fsl import FLIRT
from nipype.interfaces.fsl.maths import MathsCommand, MultiImageMaths