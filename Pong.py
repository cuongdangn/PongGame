#!/usr/bin/python3
import pygame, sys, random
from pygame.locals import *
from fonts import *


import threading
import time

BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
RED 		= (205, 92, 92)
WIDTH       = 700
HEIGHT      = 500
BALL_RADIUS = 8
PAD_WIDTH   = 8
PAD_HEIGHT  = 70
DIFF 		= 2 		# positive integer representing difficulty level
TOPSCORE    = 10	

pygame.init()
fpsClock = pygame.time.Clock()
window_obj = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption("Pong")

input_array = []

class Ball():
	def __init__(self, radius, color):
		self.radius = radius
		self.color  = color
	def reset(self, direction):
		self.pos = [int(WIDTH/2),int(HEIGHT/2)]
		if direction == "Right":
			self.vel = [random.randint(2,4),-random.randint(1,3)]
		elif direction == "Left":
			self.vel = [-random.randint(2,4),-random.randint(1,3)]
	def draw(self, window_obj):
		pygame.draw.circle(window_obj, self.color, self.pos, self.radius)
	def update(self):
		if self.pos[1] <= BALL_RADIUS or self.pos[1] >= HEIGHT-BALL_RADIUS:			# If the ball hits walls
			self.vel[1] = -self.vel[1]
		self.pos[0] += self.vel[0]
		self.pos[1] += self.vel[1]	

class Paddle():
	def __init__(self, pos, vel, width, height, color, score):
		self.pos    = pos
		self.vel    = vel
		self.width  = width
		self.height = height
		self.color  = color
		self.score  = score
	def draw(self, window_obj):
		pygame.draw.line(window_obj, self.color, (self.pos[0], self.pos[1] - self.height/2), (self.pos[0], self.pos[1] + self.height/2), self.width)
	def update(self):
		if self.pos[1]+self.vel > int(self.height/2) and self.pos[1]+self.vel < int(HEIGHT-self.height/2):
			self.pos[1]=self.pos[1]+self.vel 
	
def check_collision(ball, paddle1, paddle2):
	if ball.pos[0] <= paddle1.width*2 + ball.radius or ball.pos[0] >= WIDTH - paddle1.width*2 - ball.radius:
		if (ball.pos[1] in range(paddle1.pos[1] - int(paddle1.height/2), paddle1.pos[1] + int(paddle1.height/2)) and ball.pos[0] > int(WIDTH/2)) or (ball.pos[1] in range(paddle2.pos[1] - int(paddle2.height/2), paddle2.pos[1] + int(paddle2.height/2)) and ball.pos[0] < int(WIDTH/2)):
			if ball.vel[0]<0:
				ball.vel[0] =- (ball.vel[0]-1)
			else:
				ball.vel[0] =- (ball.vel[0]+1)
		else:
			if ball.pos[0]>int(WIDTH/2):		#check which player gets a point depending on which side the ball is in
				paddle1.score +=1
				ball.reset("Left")
			else:
				paddle2.score +=1
				ball.reset("Right")

def win_msg(window_obj, player):
	window_obj.fill(BLACK)
	try:
		font             = pygame.font.Font("fonts/Megadeth.ttf", 70)
	except:
		font             = pygame.font.Font(None, 70)
	msg 			= font.render(player + " Wins", True, WHITE)
	msgRect         = msg.get_rect()
	msgRect.centerx = int(WIDTH/2)
	msgRect.centery = int(HEIGHT/2)
	window_obj.blit(msg, msgRect)
	pygame.display.update()
	while True:	
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
			elif event.type == KEYDOWN:
				new_game(window_obj)
				return

def draw_scores(window_obj, score1, score2):
	try:
		font          = pygame.font.Font("fonts/impact.ttf", 40)
	except:
		font          = pygame.font.Font(None, 40)
	msg1          = font.render(str(score1), True, WHITE)
	msg2          = font.render(str(score2), True, WHITE)	
	msg1Rect      = msg1.get_rect()
	msg2Rect      = msg2.get_rect()
	msg1Rect.left = int(WIDTH/4)
	msg2Rect.left = int(WIDTH/4 * 3)
	msg1Rect.top  = int(HEIGHT/4)
	msg2Rect.top  = int(HEIGHT/4)
	window_obj.blit(msg1, msg1Rect)
	window_obj.blit(msg2, msg2Rect)

def draw_options(window_obj, highlight):
	try:
		font             = pygame.font.Font("fonts/Megadeth.ttf", 56)
		selected_font	 = pygame.font.Font("fonts/Megadeth.ttf", 70)
		img = pygame.image.load('movement.png')
		img.convert()
		window_obj.blit(img, [int(WIDTH*3/9), int(HEIGHT*7/10)])
	except:
		font             = pygame.font.Font(None, 56)
		selected_font	 = pygame.font.Font(None, 70)
	opt_list	  	 = [font, selected_font]
	msg1             = opt_list[highlight].render("One-Player Game", True, RED)
	msg2             = opt_list[highlight - 1].render("Two-Player Game", True, RED)	
	msg1Rect         = msg1.get_rect()
	msg2Rect         = msg2.get_rect()
	msg1Rect.centerx = int(WIDTH/2)
	msg2Rect.centerx = int(WIDTH/2)
	msg1Rect.centery = int(HEIGHT * 3/10)
	msg2Rect.centery = int(HEIGHT * 5/10)
	window_obj.blit(msg1, msg1Rect)
	window_obj.blit(msg2, msg2Rect)

def run_game(window_obj, ball, paddle1, paddle2):
	#pygame.draw.line(window, WHITE, (int(WIDTH/2), 0), (int(WIDTH/2), HEIGHT), 2)	#Draw central line and borders
	#pygame.draw.line(window, WHITE, (PAD_WIDTH, 0), (PAD_WIDTH, HEIGHT), 2)
	#pygame.draw.line(window, WHITE, (WIDTH - PAD_WIDTH, 0), (WIDTH - PAD_WIDTH, HEIGHT), 2)
	ball.update()
	ball.draw(window_obj)
	paddle1.update()
	paddle2.update()
	paddle1.draw(window_obj)
	paddle2.draw(window_obj)
	check_collision(ball, paddle1, paddle2)
	draw_scores(window_obj, paddle1.score, paddle2.score)
	
def new_game(window_obj = window_obj):
	try:
		pygame.mixer.music.load('music/cautious-path-01.mp3')
		pygame.mixer.music.play(-1)	
	except :
		pass
	ball      = Ball(BALL_RADIUS, RED)
	ball.reset("Left")
	paddle1   = Paddle([int(WIDTH-PAD_WIDTH*2),int(HEIGHT/2)], 0, PAD_WIDTH, PAD_HEIGHT, WHITE, 0)
	paddle2   = Paddle([int(PAD_WIDTH*2),int(HEIGHT/2)], 0, PAD_WIDTH, PAD_HEIGHT, WHITE, 0)
	option    = "nil"	
	highlight = 1


	while option == "nil":
		window_obj.fill(BLACK)
		draw_options(window_obj, highlight)
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
			elif event.type == KEYDOWN:
				if event.key in [K_UP, K_DOWN, K_w, K_s]:
					highlight = highlight ^ 1			#toggle between 1 and 0
				elif event.key == K_RETURN:
					option = bool(highlight ^ 1)
			pygame.display.update()
			fpsClock.tick(60)						#run at 60 fps
	game_loop(window_obj, ball, paddle1, paddle2, option)


def read_data():
	f = open('command.txt','r')
	lines = f.readlines()
	if(len(lines) == 0):
		return 'X'
	f.close()
	if(lines[-1] == 'u'):
		return 'UP'
	if(lines[-1] == 'd'):
		return 'DOWN'

def game_loop(window_obj, ball, paddle1, paddle2, two_player):
	try:
		pygame.mixer.music.stop()
		pygame.mixer.music.load('music/barn-beat-01.mp3')
		pygame.mixer.music.play(-1)	
	except :
		pass		
	while True:												#main game loop
			window_obj.fill(BLACK)							#clear screen before drawing again
			run_game(window_obj, ball, paddle1, paddle2)
			if paddle1.score >= TOPSCORE:
				win_msg(window_obj, 'Player 1')
				return
			elif paddle2.score >= TOPSCORE:
				win_msg(window_obj, 'Player 2')
				return
			
			command = read_data()
			if(command == 'UP'):
				print('UP')
				paddle1.vel = -3
			else:
				if(command == 'DOWN'):
					print('DOWN')
					paddle1.vel = 3
				else:
					paddle1.vel = 0

			for event in pygame.event.get():			#event handler
				if event.type == QUIT:
					pygame.quit()
					sys.exit()
				elif event.type == KEYDOWN:	
					if event.key == K_ESCAPE:
						new_game(window_obj)
						return
					elif event.key == K_UP:
						paddle1.vel = -3
					elif event.key == K_DOWN:
						paddle1.vel = 3				
				elif event.type == KEYUP:
					if event.key == K_UP:
						paddle1.vel = 0
					elif event.key == K_DOWN:
						paddle1.vel = 0

				if two_player:
					if event.type == KEYDOWN:				
						if event.key == K_w:
							paddle2.vel = -3
						elif event.key == K_s:
							paddle2.vel = 3				
					elif event.type == KEYUP:
						if event.key == K_w:
							paddle2.vel = 0
						elif event.key == K_s:
							paddle2.vel = 0

			if not two_player:		# AI part
				# If ball is moving away from paddle, center bat
				if ball.vel[0] > 0:
					if paddle2.pos[1] < int(HEIGHT/2):
						paddle2.vel = DIFF
					elif paddle2.pos[1] > int(HEIGHT/2):
						paddle2.vel = -DIFF					
				# if ball moving towards paddle, track its movement.
				elif ball.vel[0] < 0:
					if paddle2.pos[1] < ball.pos[1]:
						paddle2.vel = DIFF
					else :
						paddle2.vel = -DIFF
			pygame.display.update()
			fpsClock.tick(60)						#run at 60 fps


new_game()


      



# t = time.time()
# t1 = threading.Thread(target=new_game)
#     #t2 = threading.Thread(target=write_data)

# t1.start()
#     # t2.start()

#     # t1.join()
#     # t2.join()

