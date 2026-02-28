import sys
version = sys.version
with open('python_version.txt', 'w') as file:
    file.write(version)