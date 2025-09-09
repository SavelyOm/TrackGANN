from setuptools import setup, find_packages


setup(
    name="TrackGANN",
    version="0.1.0",
    author="Omelianchul Savelii",
    author_email="savelii@jinr.ru",
    description="NN for classifying tracks by events in the SPD experiment",
    url="https://github.com/SavelyOm/TrackGANN",
    packages=find_packages(),
        install_requires=[
        'torch>=2.7.1',
        'yarl>=1.18.3',
        'numpy>=1.26.4',
        'pandas>=2.2.3',
        'typer>=0.16.1',
        'tqdm>=4.67.1',
        'matplotlib>=3.10.0',
        'networkx>=3.4.2',
        'torch-geometric>=2.6.1',
        'scikit-learn>=1.6.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: CC BY-NC-ND 4.0 License',  
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific :: Artificial Intelligence/ High Energy Physic',
    ],
    python_requires=">=3.6",
)
