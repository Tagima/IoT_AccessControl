#!/bin/sh
#!/bin/bash

# Script manager for face detection and recognition algorithm
# By Tagima
# 10-2019

VERSAO=1.0
UPDATE=10_19

# ============================================================
# FUNCTIONS
# ------------------------------------------------------------



# ============================================================
# MAIN PROGRAM
# ------------------------------------------------------------

echo "+-----------------------------------------+"
echo "|               MANAGER APP               |"
echo "|            For access control           |"
echo "|                                         |"
echo "| Version: $VERSAO                            |"
echo "| Last update: $UPDATE                      |"
echo "+-----------------------------------------+"
echo ""

echo "-------------------------------------"
echo "Performing face detection"
echo "-------------------------------------"
cd face_detection_app/
python3.7 face_processing.py

if [ $? -ne 0 ] ; then
	echo "Something went wrong on face detection."
	exit 1
fi

sleep 2		# Just in case

echo ""
echo "-------------------------------------"
echo "Performing face recognition"
echo "-------------------------------------"
cd ..
cd face_recognition_app/
FACE_RES=$(python3.7 run_model_threshold.py 2>&1 | tee /dev/tty)

if [ $? -ne 0 ] ; then
	echo "Something went wrong on face recognition."
	exit 1
fi

FACE_RES=$(echo $FACE_RES | awk '{ print $NF }')	# Getting result

if [ "$FACE_RES" = "Unknown" ]; then
	# If the face is unknown, there's no need to send the information to the server

	echo "-------------------------------------"
	echo "Evaluating permission"
	echo "-------------------------------------"

	echo ""
	echo "ACCESS DENIED"
	exit 0

fi

sleep 1

echo "-------------------------------------"
echo "Performing LoRa sender"
echo "-------------------------------------"
cd ..
cd dragino_lora_app/
./dragino_lora_app sender $FACE_RES &
PID=$(echo $!)

sleep 3

kill $PID

echo ""
echo "-------------------------------------"
echo "Performing LoRa receiver"
echo "-------------------------------------"
./dragino_lora_app receiver &
PID=$(echo $!)

sleep 3

kill $PID

echo ""
echo "-------------------------------------"
echo "Evaluating permission"
echo "-------------------------------------"

FACE_PER=$(cat /var/tmp_app/message.in)

case $FACE_PER in

	DENIED )
		echo ""
		echo "ACCESS DENIED" ;;

	GRANTED )
		echo ""
		echo "ACCESS GRANTED"
		
		# TODO: Take action for opening the door here

		;;

	* )
		echo ""
		echo "UNEXPECTED SITUATION" ;;
		
esac

exit 0
