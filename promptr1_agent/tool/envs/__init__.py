def _default_env(name):
    if name == "nous":
        from promptr1_agent.tool.envs.nous import NousToolEnv
        return NousToolEnv
    elif name == "retool":
        from promptr1_agent.tool.envs.retool import ReToolEnv
        return ReToolEnv
    else:
        raise NotImplementedError(f"Tool environment {name} is not implemented")