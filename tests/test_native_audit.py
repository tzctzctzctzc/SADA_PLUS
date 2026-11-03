def test_audit_and_native_protocol_import():
    from tools.audit_native_joint import MODES,TASKS
    from tools.run_native_joint import SPEC
    assert MODES==('frozen','joint') and len(TASKS)==4 and SPEC['epochs']==5
