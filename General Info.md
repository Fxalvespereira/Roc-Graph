# Roc-Graph



Make sure to install libraires

How to Use

1. Put the Excel File in input folder
2. Put in the desired range of Crit Values (recommend to copy and paste from Excel)
3. Go into VScode and run the program
4. Wait for the program to finish 
5. Go to the output folder and check the file



Make sure the Excel file in the input follows the format of "Example Input"



There will be 2 Excel files outputted per input

1. File 1 reprocesses the input file to be software friendly 
2. Analyzed the file that contains all the data, ROC Curves, and AUC Summary



Analyzed File

Tab 1: contains original data transferred over

Tab 2: contains confusion matrices for all crit values

&nbsp;		\*Crit values are inputted in "Crit Value #" files

&nbsp;		\*Crit Values should always start with 0 and end at 10000000000 to ensure there are 2 points to act as an extreme 

&nbsp;		\*Crit Values are applied in order. Crit value 1 only applies to channel 1 in the original Excel file and only goes 		 up to 4 selected channels

Tab 3: contains the false positive rate (x-axis), true positive rate (y-axis), ROC curved based on selected Critical values, and ROC curve based on 

Tab 4: contains the area under the curve of all channels





