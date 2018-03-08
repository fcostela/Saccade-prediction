PredictScd_Online.m

Input:
	KnownSaccade - the coordinates of previous saccade positions, KnownSaccade should include at least one point, and this function can work well when the point number is larger than 2
	DelayTime - the delay time of saccade prediction, the default value is 10

Output:
	PreSaccade - the position of the predicted saccade position


Instruction: 
	There are two input parameters for the function PredictScd(KnownSaccade, DelayTime), which is used to predict the saccade position in DelayTime ms later.
	The user can input all the known saccade positions, and the function will choose the latest several positions to predict the position. The minimum length of KnownSaccade is 1. If KnownSaccade is empty, the function will return (0, 0) as the predicted position.
	The parameter DelayTime indicates the time interval between the current position and the predicted position, so that the minimum value of DelayTime is 1 ms.