
import pigpio
import time

MIN_PWM = 500
MAX_PWM = 2500

SERVO_PIN = 23
PI = None

def init_motors():
	global PI
	PI = pigpio.pi()
	PI.set_mode(SERVO_PIN, pigpio.OUTPUT)
	
def angle_to_pulse_width(angle):
    assert(angle >= 1 and angle <= 175)
    angle_norm = angle / 180.0
    pwm_range = MAX_PWM - MIN_PWM
    return pwm_range * angle_norm + MIN_PWM

"""set the target angle of the servo"""
def set_servo_angle(angle):
	global PI
	PI.set_servo_pulsewidth(SERVO_PIN, angle_to_pulse_width(angle))

init_motors()

#print(angle_to_pulse_width(0))
#print(angle_to_pulse_width(180))

set_servo_angle(90)
#for i in range(1, 175):
#	set_servo_angle(i)
#	time.sleep(.1)


PI.stop()
