from setuptools import setup, find_packages
from pkg_resources import parse_requirements
import pathlib

DISTNAME = 'asone'
DESCRIPTION = ''
MAINTAINER = 'AxcelerateAI'
MAINTAINER_EMAIL = 'umair.imran@axcelerate.ai'
URL = 'https://github.com/axcelerateai/asone'
DOWNLOAD_URL = URL

VERSION = '0.3.2'

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
        packages=find_packages(),

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
