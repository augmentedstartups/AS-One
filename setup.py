from pkg_resources import parse_requirements
import pathlib

DISTNAME = 'asone'
DESCRIPTION = ''
MAINTAINER = 'AxcelerateAI'
MAINTAINER_EMAIL = ''
URL = 'https://github.com/axcelerateai/asone-library'
DOWNLOAD_URL = URL

VERSION = '0.1-dev'

from setuptools import setup


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
        package_dir={"":'asone-linux/code'},
        packages=[DISTNAME,
                    f'{DISTNAME}.detectors',
                    f'{DISTNAME}.detectors.yolov5',
                    f'{DISTNAME}.detectors.yolov7',
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
        dependency_links = [
        "https://download.pytorch.org/whl/cu113/",
        'https://pypi.python.org/simple/'],
        install_requires=[str(requirement) for requirement in parse_requirements(requirements_txt)],

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