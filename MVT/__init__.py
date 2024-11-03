#from termcolor import colored

try:
    from MVT.backbone.point_net_pp import PointNetPP
except ImportError:
    PointNetPP = None
    msg = 'Pnet++ is not found. Hence you cannot run all models. Install it via external_tools (see README.txt there).'
    print(msg)
