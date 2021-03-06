# Copyright @ 2019 Alibaba. All rights reserved.
# Created by ruhuan on 2019.08.31
""" build wheel tool """
from __future__ import print_function
import os
import platform
IS_WINDOWS = (platform.system() == 'Windows')
IS_DARWIN = (platform.system() == 'Darwin')
IS_LINUX = (platform.system() == 'Linux')
if __name__ == '__main__':
    if IS_DARWIN:
        os.system('rm -rf build')
        os.system('python setup.py bdist_wheel')
    if IS_LINUX:
        os.system('rm -rf build')
        os.system('python setup.py bdist_wheel --plat-name=manylinux1_x86_64')
