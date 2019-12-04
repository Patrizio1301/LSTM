import inspect

def classFactory(name, argnames, baseClass, dictionary):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if key not in argnames:
                raise TypeError("Argument %s not valid for %s"
                                % (key, self.__class__.__name__))
            setattr(self, key, value)
        baseClass.__init__(self, **dictionary)

    newclass = type(name, (baseClass,), {"__init__": __init__})
    return newclass

def args(frame):
    args, _, _, values = inspect.getargvalues(frame)
    return {i:values[i] for i in args}

def regularizations(baseClass: object, arguments, kargs):
    return classFactory(baseClass.__name__, list(kargs.keys()), baseClass, arguments)



