import sys

def error_message_detail(error, error_detail=sys):
    """Extracts detailed information about the error."""
    _, _, exc_tb = error_detail.exc_info()

    # Fallback in case there is no traceback available
    if exc_tb is None:
        return str(error)
    
    # Dig into the traceback to get the file, line, and function name
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    function_name = exc_tb.tb_frame.f_code.co_name

    error_message = (
        f"Error occurred in script: [{file_name}] "
        f"at line: [{line_number}] "
        f"in function: [{function_name}()] "
        f"with error message: [{str(error)}]"
    )
    
    return error_message

class CustomException(Exception):
    """Custom exception class for detailed error tracking."""
    
    # Note: We keep error_detail=sys as a default so you don't HAVE to pass it every time
    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message