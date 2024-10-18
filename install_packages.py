import subprocess
import sys

def install_packages(requirements_file):
    try:
        with open(requirements_file, 'r') as file:
            packages = file.readlines()

        for package in packages:
            package = package.strip() 
            if package:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} installed successfully!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    install_packages('requirements.txt')
