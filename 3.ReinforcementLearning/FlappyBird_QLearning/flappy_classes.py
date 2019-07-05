import pygame
import numpy as np


class Flappy:

	def __init__(self, screen_size, flappy_pic_path):
		self._xbackground = screen_size[0]
		self._ybackground = screen_size[1]
		self._flappy_pic = pygame.image.load(flappy_pic_path)
		self._xflappy = self._flappy_pic.get_rect().size[0]
		self._yflappy = self._flappy_pic.get_rect().size[1]
		self._xposinit = int(self._xbackground/4)
		self._yposinit = int(self._ybackground/3)
		self._rotation = 0.
		self._xposition = self._xposinit
		self._yposition = self._yposinit
		self._yvelocity = 0.
		self._yvelmin = -4.
		self._yacceleration = 0.06
		self._dead = True


	def get_pos(self):

		return (self._xposition, self._yposition)


	def get_pic(self):

		return self._flappy_pic
		#return pygame.transform.rotate(self._flappy_pic, self._rotation)


	def is_dead(self):

		return self._dead


	def new_flappy(self):
		self._yvelocity = 0
		self._yposition = self._yposinit
		self._dead = False


	def actualize(self):
		# Actualize:
		self._yvelocity += self._yacceleration
		self._yposition += self._yvelocity
		# If out of screen:
		if self._yposition + self._yflappy >= self._ybackground:
			self._yposition = self._ybackground - self._yflappy
			self._dead = True
		elif self._yposition <= 0:
			self._yposition = 0
			self._dead = True
		# Rotation:
		if self._yvelocity <= 0:
			self._rotation = int(45 * self._yvelocity/self._yvelmin)
		else:
			self._rotation = -np.min([int(90 * self._yvelocity/6), 90])


	def pressed_key(self):
		self._yvelocity -= 5 * (1 + np.exp(2*self._yposition/self._ybackground-2) - np.exp(2*0.3-2))
		if self._yvelocity < self._yvelmin:
			self._yvelocity = self._yvelmin


	# def check_collision(self, pipes):
	# 	collide = list()
	# 	for pipe in pipes:
	# 		if not pipe.is_dead():
	# 			pipe1, pipe2 = pipe.get_fullbox()
	# 			collide.append(pygame.Rect(*pipe1))
	# 			collide.append(pygame.Rect(*pipe2))
	# 	flappyRect = pygame.Rect(self._xposition, self._xposition+self._xflappy, self._yposition, self._yposition+self._yflappy)
	# 	if flappyRect.collidelist(collide):
	# 		self._dead = True			

	def check_collision(self, pipes):
		try: rect1, hm1 = self._flappy_pic.rect, self._flappy_pic.hitmask
		except AttributeError: pass
		for pipe in pipes:
			try: rect2, hm2 = pipe._pipe_pic.rect, pipe._pipe_pic.hitmask
			except AttributeError: continue
			rect = rect1.clip(rect2)
			if rect.width == 0 or rect.height == 0:
				continue
			x1, y1, x2, y2 = rect.x-rect1.x, rect.y-rect1.y, rect.x-rect2.x ,rect.y-rect2.y
			for x in xrange(rect.width):
				for y in xrange(rect.height):
					if hm1[x1+x][y1+y] and hm2[x2+x][y2+y]: self._dead = True
					else: continue




class Pipe:

	def __init__(self, screen_size, pipe_pic_path):
		self._xbackground = screen_size[0]
		self._ybackground = screen_size[1]
		self._pipe_pic = pygame.image.load(pipe_pic_path)
		self._xpipe = self._pipe_pic.get_rect().size[0]
		self._ypipe = self._pipe_pic.get_rect().size[1]
		self._gap = 100
		self._xposition = self._xbackground
		self._yposition = np.random.rand() * (int(self._ybackground)-int(self._ybackground/5)-self._gap) + (int(self._ybackground/10)+self._gap)
		self._xvelocity = 2
		self._dead = True


	def get_pos(self):

		return (self._xposition, self._yposition), (self._xposition, self._yposition-self._gap-self._ypipe)


	def get_fullbox(self):

		return (int(self._xposition), int(self._xposition+self._xpipe), int(self._yposition), int(self._yposition+self._ypipe)), (int(self._xposition), int(self._xposition+self._xpipe), int(self._yposition-self._gap-self._ypipe), int(self._yposition-self._gap))


	def get_pic(self):

		return self._pipe_pic, pygame.transform.rotate(self._pipe_pic, 180)


	def is_dead(self):

		return self._dead


	def new_pipe(self):
		self._dead = False


	def actualize(self):
		self._xposition -= self._xvelocity
		# Kill pipe if out of screen
		if self._xposition - self._xpipe < 0:
			self._dead = True
