class ExperimentSettings:
    def __init__(self, settings_file):
        self.read_from_file(settings_file)

    def read_from_file(self, path):
        f = open(path, 'r')
        for l in f.readlines():
            key,val = l.split(':')
            if val.endswith('\n'):
            	val = val[:-1]         
            self.__dict__[key] = val 

    def to_string(self):
    	return str(self.__dict__)  