from configparser import ConfigParser


class MyConfigParser:
    def __init__(self, section):
        self.section = section
        self.config = ConfigParser()
        try:
            self.config.read("./config.ini")
        except IOError:
            print("File config.ini doesn't exist!")

    def config_section_map(self):
        dict1 = {}
        options = self.config.options(self.section)
        for option in options:
            try:
                dict1[option] = self.config.get(self.section, option)
            except:
                dict1[option] = None
        return dict1
