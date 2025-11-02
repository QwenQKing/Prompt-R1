def _default_tool(name):
    print("***********Prompt-R1*************")
    if name == "prompt":
        from agent_r1.tool.tools.LLM_tool import LLM_Tool
        return LLM_Tool()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")