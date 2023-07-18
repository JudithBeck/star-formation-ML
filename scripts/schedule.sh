#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py trainer.max_epochs=5 logger=csv

python src/train.py trainer.max_epochs=10 logger=csv


#In the folder 'scripts', you have a bash script named 'schedule.sh'. This script is designed to schedule the execution of two training runs using the 'train.py' script with different configurations. The script specifies two separate training runs with different values for the 'trainer.max_epochs' and 'logger' configurations.

#Here's the breakdown of the script:

#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

# The shebang '#!/bin/bash' indicates that this is a Bash script.

# The following lines are comments providing information about the script.
# The comments start with '#'.

# The script is scheduled to be run from the root folder with the command 'bash scripts/schedule.sh'.
# This assumes that the 'train.py' script is located inside the 'src' folder.

# The two training runs are specified using the 'train.py' script with different configurations.

# First Training Run:
# The 'trainer.max_epochs' configuration is set to 5, which means the training will run for 5 epochs.
# The 'logger' is set to 'csv', indicating that the training metrics will be logged in a CSV format.
#python src/train.py trainer.max_epochs=5 logger=csv

# Second Training Run:
# The 'trainer.max_epochs' configuration is set to 10, which means the training will run for 10 epochs.
# The 'logger' is set to 'csv', indicating that the training metrics will be logged in a CSV format.
#python src/train.py trainer.max_epochs=10 logger=csv


#When you run the 'schedule.sh' script using the command 'bash scripts/schedule.sh', it will execute the two 'train.py' commands in sequence, triggering two separate training runs with different configurations as specified in the script.