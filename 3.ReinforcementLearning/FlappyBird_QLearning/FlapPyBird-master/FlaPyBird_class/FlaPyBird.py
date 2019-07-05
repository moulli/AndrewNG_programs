from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *


class FlaPyBird:
	def __init__(self):
		self.SCREENWIDTH = 288
		self.SCREENHEIGHT = 512
		pygame.init()
		SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
		pygame.display.set_caption('Flappy Bird')
		self.PIPEGAPSIZE = 100
		self.BASEY = self.SCREENHEIGHT * 0.79
		self.action_space.n = 2
		self.action_space = [0, 1]
		self.observation_space.n = [self.SCREENHEIGHT, self.SCREENWIDTH]
		# list of all possible players (tuple of 3 positions of flap)
		self.PLAYERS_LIST = (
			# red bird
			(
				'assets/sprites/redbird-upflap.png',
				'assets/sprites/redbird-midflap.png',
				'assets/sprites/redbird-downflap.png',
			),
			# blue bird
			(
				'assets/sprites/bluebird-upflap.png',
				'assets/sprites/bluebird-midflap.png',
				'assets/sprites/bluebird-downflap.png',
			),
			# yellow bird
			(
				'assets/sprites/yellowbird-upflap.png',
				'assets/sprites/yellowbird-midflap.png',
				'assets/sprites/yellowbird-downflap.png',
			),
		)
		# list of backgrounds
		self.BACKGROUNDS_LIST = (
			'assets/sprites/background-day.png',
			'assets/sprites/background-night.png',
		)
		# list of pipes
		self.PIPES_LIST = (
			'assets/sprites/pipe-green.png',
			'assets/sprites/pipe-red.png',
		)
		self.IMAGES = {}
		randBg = random.randint(0, len(self.BACKGROUNDS_LIST) - 1)
		self.IMAGES['background'] = pygame.image.load(self.BACKGROUNDS_LIST[randBg]).convert()
		randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
		self.IMAGES['player'] = (
			pygame.image.load(self.PLAYERS_LIST[randPlayer][0]).convert_alpha(),
			pygame.image.load(self.PLAYERS_LIST[randPlayer][1]).convert_alpha(),
			pygame.image.load(self.PLAYERS_LIST[randPlayer][2]).convert_alpha(),
		)
		pipeindex = random.randint(0, len(self.PIPES_LIST) - 1)
		self.IMAGES['pipe'] = (
			pygame.transform.flip(
				pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(), False, True),
			pygame.image.load(self.PIPES_LIST[pipeindex]).convert_alpha(),
		)
		self.HITMASKS = {}
		self.HITMASKS['pipe'] = (
			getHitmask(self.IMAGES['pipe'][0]),
			getHitmask(self.IMAGES['pipe'][1]),
		)
		self.HITMASKS['player'] = (
			getHitmask(self.IMAGES['player'][0]),
			getHitmask(self.IMAGES['player'][1]),
			getHitmask(self.IMAGES['player'][2]),
		)

	def reset(self):
		pass

	def render(self):
		pass

	def step(self, action): 
		pass

	def close(self):
		pygame.display.quit()
		pygame.quit()

	def getHitmask(image):
		"""returns a hitmask using an image's alpha."""
		mask = []
		for x in xrange(image.get_width()):
			mask.append([])
			for y in xrange(image.get_height()):
				mask[x].append(bool(image.get_at((x,y))[3]))
		return mask