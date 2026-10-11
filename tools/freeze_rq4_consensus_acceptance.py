"""核验真实共识诊断并绑定当前clean代码；不启动训练。"""
import argparse
from tools.rq4_consensus import freeze_acceptance

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--reports',nargs='+',required=True)
    freeze_acceptance(**vars(parser.parse_args()))
