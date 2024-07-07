import re

class Config:
    """
    A simple class to hold configuration options as attributes. """
    pass

def parse_opt_file(file_path):
    """
    Parse an option file and return a Config object with the options as attributes.
    """
    opt = Config()

    with open(file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        # Ignore lines that do not match the key: value pattern
        if not re.match(r'^\s*[^#]+:\s*[^#]+', line):
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().split()[0]  # Only take the first part before any comments

        try:
            # Try to convert value to a number (int or float), otherwise keep as string
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass

        setattr(opt, key, value)  # Set the attribute on the opt object

    return opt
