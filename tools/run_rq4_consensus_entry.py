"""新共识方法的独立入口；默认正式启动必须有匹配的真实验收凭证。"""
from tools.run_rq4_entry import main

if __name__=='__main__': main(consensus=True)
