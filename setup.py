from setuptools import find_packages,setup

Hypen_e='-e .'

def get_requirements(file_path):
    """
    This function will return the list of packages
    """

    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]
        if Hypen_e in requirements:
            requirements.remove(Hypen_e)
    return requirements


setup(
    name="Insurance_Premium_Prediction",
    author="Karan Mehta",
    version='0.0.0.1',
    author_email='kmehta883@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)