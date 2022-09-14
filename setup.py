from setuptools import setup
from pkg_resources import parse_requirements
import pathlib

DISTNAME = 'asone'
DESCRIPTION = ''
MAINTAINER = 'AxcelerateAI'
MAINTAINER_EMAIL = ''
URL = 'https://github.com/axcelerateai/asone-library'
DOWNLOAD_URL = URL

VERSION = '0.1-dev'


requirements_txt = pathlib.Path('requirements.txt').open()


def setup_package():
    setup(
        name=DISTNAME,
        version=VERSION,
        description=DESCRIPTION,
        url=DOWNLOAD_URL,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        license='BSD 2-clause',
        # package_dir={"":""},
        packages=[DISTNAME,
                  f'{DISTNAME}.detectors',
                  f'{DISTNAME}.detectors.utils',
                  f'{DISTNAME}.detectors.yolov5',
                  f'{DISTNAME}.detectors.yolov5.yolov5.models',
                  f'{DISTNAME}.detectors.yolov5.yolov5.utils',
                  f'{DISTNAME}.detectors.yolov6',
                  f'{DISTNAME}.detectors.yolov6.yolov6.layers',
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
                  f'{DISTNAME}.detectors.yolox.exp.default',
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
            'License :: OSI Approved :: BSD License',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
        ],
    )


if __name__ == "__main__":
    setup_package()
