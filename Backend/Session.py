try:
    from streamlit.scriptrunner.script_run_context import get_script_run_ctx
except ModuleNotFoundError:
    # streamlit < 1.4
    from streamlit.report_thread import (  # type: ignore
        get_report_ctx as get_script_run_ctx,
    )

            
from streamlit.server.server import Server

            
# Ref: https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92

            

            
def get_session_id() -> str:
    ctx = get_script_run_ctx()
    if ctx is None:
        raise Exception("Failed to get the thread context")

            
    return ctx.session_id
