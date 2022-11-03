from setuptools import setup
from pkg_resources import parse_requirements
import pathlib

DISTNAME = 'asone'
DESCRIPTION = ''
MAINTAINER = 'AxcelerateAI'
MAINTAINER_EMAIL = ''
URL = 'https://github.com/axcelerateai/asone'
DOWNLOAD_URL = URL

VERSION = '0.1.2.dev10'

with open('README.md') as f:
    long_description = f.read()

requirements_txt = pathlib.Path('requirements.txt').open()


def setup_package():
    setup(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description = long_description,
        long_description_content_type='text/markdown',
        url=DOWNLOAD_URL,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        license='BSD 2-clause',
        keywords='asone bytetrack deepsort norfair yolo yolox yolor yolov5 yolov7 installation inferencing',
        # package_dir={"":""},
        packages=[DISTNAME,
                  f'{DISTNAME}.detectors',
                  f'{DISTNAME}.detectors.utils',
                  f'{DISTNAME}.detectors.yolov5',
                  f'{DISTNAME}.detectors.yolov5.yolov5.models',
                  f'{DISTNAME}.detectors.yolov5.yolov5.utils',
                  f'{DISTNAME}.detectors.yolov6',
                  f'{DISTNAME}.detectors.yolov6.yolov6.layers',
                  f'{DISTNAME}.detectors.yolov6.yolov6.assigners',
                  f'{DISTNAME}.detectors.yolov6.yolov6.models',
                  f'{DISTNAME}.detectors.yolov6.yolov6.utils',
                  f'{DISTNAME}.detectors.yolov7',
                  f'{DISTNAME}.detectors.yolov7.yolov7.models',
                  f'{DISTNAME}.detectors.yolov7.yolov7.utils',
                  f'{DISTNAME}.detectors.yolor',
                  f'{DISTNAME}.detectors.yolor.models',
                  f'{DISTNAME}.detectors.yolor.utils',
                  f'{DISTNAME}.detectors.yolor.cfg',
                  f'{DISTNAME}.detectors.yolox',
                  f'{DISTNAME}.detectors.yolox.exps',
                  f'{DISTNAME}.detectors.yolox.yolox',
                  f'{DISTNAME}.detectors.yolox.yolox.exp',
                  f'{DISTNAME}.detectors.yolox.yolox.models',
                  f'{DISTNAME}.detectors.yolox.yolox.utils',
                  f'{DISTNAME}.trackers',
                  f'{DISTNAME}.trackers.byte_track',
                  f'{DISTNAME}.trackers.byte_track.tracker',
                  f'{DISTNAME}.trackers.deep_sort',
                  f'{DISTNAME}.trackers.deep_sort.tracker',
                  f'{DISTNAME}.trackers.deep_sort.tracker.deep',
                  f'{DISTNAME}.trackers.deep_sort.tracker.sort',
                  f'{DISTNAME}.trackers.nor_fair',
                  f'{DISTNAME}.utils',
                  ],

        dependency_links=[
            "https://download.pytorch.org/whl/cu113/",
            'https://pypi.python.org/simple/'],
        install_requires=[str(requirement)
                          for requirement in parse_requirements(requirements_txt)],
        package_data={
            "": ["detectors/yolor/cfg/*.cfg", "detectors/data/*.yaml",
                         "detectors/data/*.yml", "detectors/data/*.names"],
        },

        include_package_data=True,
        classifiers=[
            'Development Status :: 1 - Planning',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Microsoft :: Windows :: Windows 10',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
        ],
    )


if __name__ == "__main__":
    setup_package()
