import sys
import logging

def error_handling():
    logging.critical(' {}. {}, line: {}'.format(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]))
    sys.exit(1)