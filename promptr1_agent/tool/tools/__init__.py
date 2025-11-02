def _default_tool(name):
    print("***********name*************:",name)
    if name == "prompt":
        from promptr1_agent.tool.tools.LLM_tool import LLM_Tool
        return LLM_Tool()
    else:
        raise NotImplementedError(f"Tool {name} not implemented")