import setuptools
import os
import sys

def get_version():
  version_path = os.path.join(os.path.dirname(__file__), 'reinforceable')
  sys.path.insert(0, version_path)
  from _version import __version__ as version
  return version

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "tensorflow==2.13.0",
    "tensorflow-probability==0.20.1",
    "gymnasium[all]==0.26.2",
]

setuptools.setup(
    name='reinforceable',
    version=get_version(),
    author="Alexander Kensert",
    author_email="alexander.kensert@gmail.com",
    description="Implementations of reinforcement learning algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/akensert/reinforceable",
    packages=setuptools.find_packages(include=["reinforceable*"]),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.10",
    keywords=[
        'tensorflow',
        'keras',
        'deep-learning',
        'machine-learning',
        'reinforcement-learning',
        'agent',
        'ppo',
        'recurrent-ppo',
        'rnn-ppo',
        'gru-ppo',
        'lstm-ppo',
    ]
)
