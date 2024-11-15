from setuptools import find_packages, setup
from typing import List


HYPEN_IN_FILE = '-e .'
def get_requirements(file_path)->List[str]:
    '''
        
    This function will return the list of requirements

    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_IN_FILE in requirements:
            requirements.remove(HYPEN_IN_FILE)
            
    
    return requirements        

setup(
    name="financeml",
    version="0.0.1",
    description="Financial Machine Learning",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)

