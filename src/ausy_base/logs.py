import logging, os


class MyLogger:

    def __init__(cls, name='test'):
        cwd = os.getcwd()
        ik = cwd.find("src")
        cls.path = cwd[:ik]
        logging.basicConfig(filename=cls.path + "log" + name,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        logging.info("P3o logging")
        cls.logger = logging.getLogger(name)