from typing import List

import pytest

from tests.helpers.package_available import _SH_AVAILABLE

if _SH_AVAILABLE:
    import sh


def run_sh_command(command: List[str]):
    """Default method for executing shell commands with pytest and sh package."""
    msg = None
    try:
        sh.python(command)
    except sh.ErrorReturnCode as e:
        msg = e.stderr.decode()
    if msg:
        pytest.fail(msg=msg)



#The code you provided defines a function run_sh_command for executing shell commands using the sh package. This function is used to run shell commands with pytest and verify the output. It is designed to handle cases where the sh package is available, and it uses the sh.python function to execute the provided shell command.

#Here's a breakdown of the code:

#The function run_sh_command takes a single argument command, which is a list of strings representing the shell command to be executed.

#The function begins by checking if the sh package is available by importing _SH_AVAILABLE from the tests.helpers.package_available module. If the sh package is not available, the function will not be defined, and any attempt to call it will raise an ImportError.

#Inside the function, there is a try-except block that attempts to execute the shell command using sh.python(command).

#If the shell command execution raises an sh.ErrorReturnCode, it means that the command returned a non-zero exit status, indicating an error. In this case, the function captures the error message from the exception's stderr attribute and stores it in the variable msg.

#Finally, if msg is not None, it means there was an error during the command execution. The function uses pytest.fail to fail the current test with the error message as the failure reason. This approach allows you to use the function within pytest test cases to assert the correct behavior of shell commands.

#Note that if the sh package is not available, the function will not be defined, and any attempt to call it will raise an NameError.

#To use this run_sh_command function in your tests, you should have the sh package installed in your environment. When the sh package is available, you can call the function to run shell commands as part of your test cases and check the expected behavior.

#Here's an example of how you can use the run_sh_command function in a test case:

#python
#Copy code
#def test_shell_command():
#    # Define the shell command to be executed
#    command = ["echo", "Hello, World!"]

    # Call the run_sh_command function to execute the shell command
    run_sh_command(command)
#In this example, the test_shell_command test case will pass if the sh package is available and the shell command "echo 'Hello, World!'" executes successfully. Otherwise, the test will fail with the appropriate error message.