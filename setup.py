from setuptools import setup, find_packages

setup(
    name='unikan', 
    version='0.2.1', 
    author='Zhijie Chen', 
    author_email='zhijiechencs@gmail.com', 
    description='A Python library based on pytorch for universally building KAN-type networks', 
    long_description=open('README.md', 'r', encoding='utf-8').read(), 
    long_description_content_type='text/markdown', 
    url='https://github.com/chikkkit/uni-kan', 
    packages=find_packages(), 
    install_requires=[
        
    ],
    classifiers=[ 
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6', 
)